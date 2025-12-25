#ifndef SAMPLER_H
#define SAMPLER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 采样器句柄
typedef void* SamplerHandle;

// 采样策略枚举
typedef enum {
    STRATEGY_GREEDY = 0,
    STRATEGY_TEMPERATURE = 1,
    STRATEGY_TOP_K = 2,
    STRATEGY_TOP_P = 3,
    STRATEGY_AUTO = 4
} SamplingStrategy;

// 性能统计结构
typedef struct {
    uint64_t total_samples;
    double total_time;
    uint64_t greedy_samples;
    uint64_t temperature_samples;
    uint64_t topk_samples;
    uint64_t topp_samples;
    double average_sample_time;
    double samples_per_second;
} SamplerStats;

// 创建采样器
SamplerHandle sampler_create(int vocab_size);

// 销毁采样器
void sampler_destroy(SamplerHandle handle);

// 主采样函数
int sampler_sample(SamplerHandle handle, 
                   const float* logits, 
                   int vocab_size,
                   float temperature, 
                   float top_p, 
                   int top_k);

// 贪婪采样
int sampler_sample_greedy(SamplerHandle handle, 
                          const float* logits, 
                          int vocab_size);

// 温度采样
int sampler_sample_temperature(SamplerHandle handle, 
                               const float* logits, 
                               int vocab_size,
                               float temperature);

// Top-K采样
int sampler_sample_topk(SamplerHandle handle, 
                        const float* logits, 
                        int vocab_size,
                        int top_k,
                        float temperature);

// Top-P采样
int sampler_sample_topp(SamplerHandle handle, 
                        const float* logits, 
                        int vocab_size,
                        float top_p,
                        float temperature);

// 获取性能统计
void sampler_get_stats(SamplerHandle handle, SamplerStats* stats);

// 重置统计
void sampler_reset_stats(SamplerHandle handle);

// 设置优化配置
void sampler_set_greedy_threshold(SamplerHandle handle, float threshold);
void sampler_set_fast_topk_threshold(SamplerHandle handle, int threshold);
void sampler_set_vocab_limit(SamplerHandle handle, int limit);

#ifdef __cplusplus
}
#endif

#endif // SAMPLER_H
