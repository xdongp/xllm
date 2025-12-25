/**
 * xLLM C语言采样器实现
 * 对应Python版本的sampler.py，提供高性能的token采样功能
 */

#include "sampler.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>

// 采样器内部结构
typedef struct {
    int vocab_size;
    
    // 性能统计
    uint64_t total_samples;
    double total_time;
    uint64_t greedy_samples;
    uint64_t temperature_samples;
    uint64_t topk_samples;
    uint64_t topp_samples;
    
    // 优化配置
    float greedy_threshold;      // 激进贪婪采样阈值
    int fast_topk_threshold;     // 快速Top-K阈值
    float temperature_threshold; // 温度采样阈值
    int vocab_limit;             // 词汇表限制
    
    // 随机数生成器
    unsigned int seed;
} SamplerImpl;

// 辅助函数：比较浮点数（用于排序）
static int compare_floats_desc(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    if (fa < fb) return 1;
    if (fa > fb) return -1;
    return 0;
}

// 辅助函数：快速排序索引
typedef struct {
    float value;
    int index;
} IndexedFloat;

static int compare_indexed_floats_desc(const void* a, const void* b) {
    const IndexedFloat* ia = (const IndexedFloat*)a;
    const IndexedFloat* ib = (const IndexedFloat*)b;
    if (ia->value < ib->value) return 1;
    if (ia->value > ib->value) return -1;
    return 0;
}

// 辅助函数：计算softmax
static void softmax(float* logits, int size) {
    // 数值稳定：减去最大值
    float max_logit = -FLT_MAX;
    for (int i = 0; i < size; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }
    
    // 计算exp并求和
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        logits[i] = expf(logits[i] - max_logit);
        sum += logits[i];
    }
    
    // 归一化
    for (int i = 0; i < size; i++) {
        logits[i] /= sum;
    }
}

// 辅助函数：从概率分布中采样
static int sample_from_distribution(const float* probs, int size, unsigned int* seed) {
    float r = (float)rand_r(seed) / (float)RAND_MAX;
    float cumulative = 0.0f;
    
    for (int i = 0; i < size; i++) {
        cumulative += probs[i];
        if (r <= cumulative) {
            return i;
        }
    }
    
    return size - 1; // 返回最后一个元素作为fallback
}

// 创建采样器
SamplerHandle sampler_create(int vocab_size) {
    SamplerImpl* sampler = (SamplerImpl*)malloc(sizeof(SamplerImpl));
    if (!sampler) {
        return NULL;
    }
    
    sampler->vocab_size = vocab_size;
    
    // 初始化性能统计
    sampler->total_samples = 0;
    sampler->total_time = 0.0;
    sampler->greedy_samples = 0;
    sampler->temperature_samples = 0;
    sampler->topk_samples = 0;
    sampler->topp_samples = 0;
    
    // 初始化优化配置（与Python版本一致）
    sampler->greedy_threshold = 0.15f;
    sampler->fast_topk_threshold = 3;
    sampler->temperature_threshold = 0.3f;
    sampler->vocab_limit = 50;
    
    // 初始化随机数种子
    sampler->seed = (unsigned int)time(NULL);
    
    return (SamplerHandle)sampler;
}

// 销毁采样器
void sampler_destroy(SamplerHandle handle) {
    if (handle) {
        free((SamplerImpl*)handle);
    }
}

// 贪婪采样 - 直接选择最大值
int sampler_sample_greedy(SamplerHandle handle,
                          const float* logits,
                          int vocab_size) {
    if (!handle || !logits || vocab_size <= 0) {
        return -1;
    }

    SamplerImpl* sampler = (SamplerImpl*)handle;

    clock_t start_time = clock();

    int max_index = 0;
    float max_value = logits[0];

    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_value) {
            max_value = logits[i];
            max_index = i;
        }
    }

    sampler->greedy_samples++;

    // 更新统计
    clock_t end_time = clock();
    double elapsed = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    sampler->total_samples++;
    sampler->total_time += elapsed;

    return max_index;
}

// 温度采样
int sampler_sample_temperature(SamplerHandle handle,
                               const float* logits,
                               int vocab_size,
                               float temperature) {
    if (!handle || !logits || vocab_size <= 0 || temperature <= 0.0f) {
        return -1;
    }

    SamplerImpl* sampler = (SamplerImpl*)handle;

    clock_t start_time = clock();

    // 复制logits以避免修改原始数据
    float* scaled_logits = (float*)malloc(vocab_size * sizeof(float));
    if (!scaled_logits) {
        return -1;
    }

    // 应用温度缩放
    for (int i = 0; i < vocab_size; i++) {
        scaled_logits[i] = logits[i] / temperature;
    }

    // 计算softmax
    softmax(scaled_logits, vocab_size);

    // 从概率分布中采样
    int token_id = sample_from_distribution(scaled_logits, vocab_size, &sampler->seed);

    free(scaled_logits);
    sampler->temperature_samples++;

    // 更新统计
    clock_t end_time = clock();
    double elapsed = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    sampler->total_samples++;
    sampler->total_time += elapsed;

    return token_id;
}

// Top-K采样
int sampler_sample_topk(SamplerHandle handle,
                        const float* logits,
                        int vocab_size,
                        int top_k,
                        float temperature) {
    if (!handle || !logits || vocab_size <= 0 || top_k <= 0) {
        return -1;
    }

    SamplerImpl* sampler = (SamplerImpl*)handle;

    clock_t start_time = clock();

    // 限制k的范围
    if (top_k > vocab_size) {
        top_k = vocab_size;
    }

    // 创建索引数组
    IndexedFloat* indexed = (IndexedFloat*)malloc(vocab_size * sizeof(IndexedFloat));
    if (!indexed) {
        return -1;
    }

    // 填充索引数组
    for (int i = 0; i < vocab_size; i++) {
        indexed[i].value = logits[i];
        indexed[i].index = i;
    }

    // 排序（降序）
    qsort(indexed, vocab_size, sizeof(IndexedFloat), compare_indexed_floats_desc);

    // 提取top-k的值
    float* top_logits = (float*)malloc(top_k * sizeof(float));
    int* top_indices = (int*)malloc(top_k * sizeof(int));
    if (!top_logits || !top_indices) {
        free(indexed);
        if (top_logits) free(top_logits);
        if (top_indices) free(top_indices);
        return -1;
    }

    for (int i = 0; i < top_k; i++) {
        top_logits[i] = indexed[i].value;
        top_indices[i] = indexed[i].index;
    }

    free(indexed);

    // 应用温度
    if (temperature != 1.0f) {
        for (int i = 0; i < top_k; i++) {
            top_logits[i] /= temperature;
        }
    }

    // 计算softmax
    softmax(top_logits, top_k);

    // 快速决策优化：如果最高概率超过90%，直接选择
    if (top_logits[0] > 0.9f) {
        int result = top_indices[0];
        free(top_logits);
        free(top_indices);
        sampler->topk_samples++;

        // 更新统计
        clock_t end_time = clock();
        double elapsed = (double)(end_time - start_time) / CLOCKS_PER_SEC;
        sampler->total_samples++;
        sampler->total_time += elapsed;

        return result;
    }

    // 否则进行采样
    int selected_idx = sample_from_distribution(top_logits, top_k, &sampler->seed);
    int result = top_indices[selected_idx];

    free(top_logits);
    free(top_indices);
    sampler->topk_samples++;

    // 更新统计
    clock_t end_time = clock();
    double elapsed = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    sampler->total_samples++;
    sampler->total_time += elapsed;

    return result;
}

// Top-P采样
int sampler_sample_topp(SamplerHandle handle,
                        const float* logits,
                        int vocab_size,
                        float top_p,
                        float temperature) {
    if (!handle || !logits || vocab_size <= 0 || top_p <= 0.0f || top_p > 1.0f) {
        return -1;
    }

    SamplerImpl* sampler = (SamplerImpl*)handle;

    clock_t start_time = clock();

    // 创建索引数组
    IndexedFloat* indexed = (IndexedFloat*)malloc(vocab_size * sizeof(IndexedFloat));
    if (!indexed) {
        return -1;
    }

    // 填充索引数组
    for (int i = 0; i < vocab_size; i++) {
        indexed[i].value = logits[i];
        indexed[i].index = i;
    }

    // 排序（降序）
    qsort(indexed, vocab_size, sizeof(IndexedFloat), compare_indexed_floats_desc);

    // 应用温度
    if (temperature != 1.0f) {
        for (int i = 0; i < vocab_size; i++) {
            indexed[i].value /= temperature;
        }
    }

    // 计算softmax
    float* probs = (float*)malloc(vocab_size * sizeof(float));
    if (!probs) {
        free(indexed);
        return -1;
    }

    // 提取值并计算softmax
    float max_logit = -FLT_MAX;
    for (int i = 0; i < vocab_size; i++) {
        if (indexed[i].value > max_logit) {
            max_logit = indexed[i].value;
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = expf(indexed[i].value - max_logit);
        sum += probs[i];
    }

    for (int i = 0; i < vocab_size; i++) {
        probs[i] /= sum;
    }

    // 计算累积概率并找到截断点
    float cumulative = 0.0f;
    int cutoff_idx = 0;
    for (int i = 0; i < vocab_size; i++) {
        cumulative += probs[i];
        if (cumulative >= top_p) {
            cutoff_idx = i + 1;
            break;
        }
    }

    // 确保至少保留一个token
    if (cutoff_idx == 0) {
        cutoff_idx = 1;
    }

    // 截断并重新归一化
    float truncated_sum = 0.0f;
    for (int i = 0; i < cutoff_idx; i++) {
        truncated_sum += probs[i];
    }

    for (int i = 0; i < cutoff_idx; i++) {
        probs[i] /= truncated_sum;
    }

    // 采样
    int selected_idx = sample_from_distribution(probs, cutoff_idx, &sampler->seed);
    int result = indexed[selected_idx].index;

    free(indexed);
    free(probs);
    sampler->topp_samples++;

    // 更新统计
    clock_t end_time = clock();
    double elapsed = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    sampler->total_samples++;
    sampler->total_time += elapsed;

    return result;
}

// 前向声明
int sampler_sample_temperature_limited(SamplerHandle handle,
                                       const float* logits,
                                       int vocab_size,
                                       float temperature);

// 主采样函数 - 自适应策略选择
int sampler_sample(SamplerHandle handle, 
                   const float* logits, 
                   int vocab_size,
                   float temperature, 
                   float top_p, 
                   int top_k) {
    if (!handle || !logits || vocab_size <= 0) {
        return -1;
    }
    
    SamplerImpl* sampler = (SamplerImpl*)handle;
    
    clock_t start_time = clock();
    
    int token_id;
    
    // 自动选择最优采样策略 - 激进优化（与Python版本一致）
    if (temperature < sampler->greedy_threshold) {
        // 激进贪婪采样 - 最快路径
        token_id = sampler_sample_greedy(handle, logits, vocab_size);
        
    } else if (top_k <= sampler->fast_topk_threshold) {
        // 超快Top-K采样 - 限制范围
        token_id = sampler_sample_topk(handle, logits, vocab_size, top_k, temperature);
        
    } else if (temperature <= sampler->temperature_threshold) {
        // 低温度快速采样 - 使用小Top-K
        int k_limited = top_k < 10 ? top_k : 10;
        token_id = sampler_sample_topk(handle, logits, vocab_size, k_limited, temperature);
        
    } else {
        // 限制词汇表的温度采样
        token_id = sampler_sample_temperature_limited(handle, logits, vocab_size, temperature);
    }
    
    // 更新统计
    clock_t end_time = clock();
    double elapsed = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    sampler->total_samples++;
    sampler->total_time += elapsed;
    
    return token_id;
}

// 限制词汇表的快速温度采样
int sampler_sample_temperature_limited(SamplerHandle handle,
                                      const float* logits,
                                      int vocab_size,
                                      float temperature) {
    if (!handle || !logits || vocab_size <= 0 || temperature <= 0.0f) {
        return -1;
    }
    
    SamplerImpl* sampler = (SamplerImpl*)handle;
    
    // 只考虑前N个最可能的token以提高速度
    int vocab_limit = sampler->vocab_limit < vocab_size ? sampler->vocab_limit : vocab_size;
    
    // 创建索引数组
    IndexedFloat* indexed = (IndexedFloat*)malloc(vocab_size * sizeof(IndexedFloat));
    if (!indexed) {
        return -1;
    }
    
    // 填充索引数组
    for (int i = 0; i < vocab_size; i++) {
        indexed[i].value = logits[i];
        indexed[i].index = i;
    }
    
    // 排序（降序）
    qsort(indexed, vocab_size, sizeof(IndexedFloat), compare_indexed_floats_desc);
    
    // 提取top-N的值
    float* top_logits = (float*)malloc(vocab_limit * sizeof(float));
    int* top_indices = (int*)malloc(vocab_limit * sizeof(int));
    if (!top_logits || !top_indices) {
        free(indexed);
        if (top_logits) free(top_logits);
        if (top_indices) free(top_indices);
        return -1;
    }
    
    for (int i = 0; i < vocab_limit; i++) {
        top_logits[i] = indexed[i].value;
        top_indices[i] = indexed[i].index;
    }
    
    free(indexed);
    
    // 应用温度缩放
    for (int i = 0; i < vocab_limit; i++) {
        top_logits[i] /= temperature;
    }
    
    // 计算softmax
    softmax(top_logits, vocab_limit);
    
    // 快速采样
    int selected_idx = sample_from_distribution(top_logits, vocab_limit, &sampler->seed);
    int result = top_indices[selected_idx];
    
    free(top_logits);
    free(top_indices);
    sampler->temperature_samples++;
    
    return result;
}

// 获取性能统计
void sampler_get_stats(SamplerHandle handle, SamplerStats* stats) {
    if (!handle || !stats) {
        return;
    }
    
    SamplerImpl* sampler = (SamplerImpl*)handle;
    
    stats->total_samples = sampler->total_samples;
    stats->total_time = sampler->total_time;
    stats->greedy_samples = sampler->greedy_samples;
    stats->temperature_samples = sampler->temperature_samples;
    stats->topk_samples = sampler->topk_samples;
    stats->topp_samples = sampler->topp_samples;
    
    if (sampler->total_samples > 0) {
        stats->average_sample_time = sampler->total_time / sampler->total_samples;
        stats->samples_per_second = 1.0 / stats->average_sample_time;
    } else {
        stats->average_sample_time = 0.0;
        stats->samples_per_second = 0.0;
    }
}

// 重置统计
void sampler_reset_stats(SamplerHandle handle) {
    if (!handle) {
        return;
    }
    
    SamplerImpl* sampler = (SamplerImpl*)handle;
    
    sampler->total_samples = 0;
    sampler->total_time = 0.0;
    sampler->greedy_samples = 0;
    sampler->temperature_samples = 0;
    sampler->topk_samples = 0;
    sampler->topp_samples = 0;
}

// 设置优化配置
void sampler_set_greedy_threshold(SamplerHandle handle, float threshold) {
    if (handle) {
        SamplerImpl* sampler = (SamplerImpl*)handle;
        sampler->greedy_threshold = threshold;
    }
}

void sampler_set_fast_topk_threshold(SamplerHandle handle, int threshold) {
    if (handle && threshold > 0) {
        SamplerImpl* sampler = (SamplerImpl*)handle;
        sampler->fast_topk_threshold = threshold;
    }
}

void sampler_set_vocab_limit(SamplerHandle handle, int limit) {
    if (handle && limit > 0) {
        SamplerImpl* sampler = (SamplerImpl*)handle;
        sampler->vocab_limit = limit;
    }
}
