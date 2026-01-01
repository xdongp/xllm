#!/usr/bin/env python3
import os

def check_markdown_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    in_code_block = False
    code_block_type = None
    code_block_start = None
    code_blocks = []
    errors = []
    
    for i, line in enumerate(lines, 1):
        if line.strip().startswith('```'):
            if not in_code_block:
                # Opening a code block
                in_code_block = True
                code_block_start = i
                code_block_type = line.strip()[3:].strip() or 'code'
            else:
                # Closing a code block
                code_blocks.append({
                    'start': code_block_start,
                    'end': i,
                    'type': code_block_type
                })
                in_code_block = False
                code_block_start = None
                code_block_type = None
    
    if in_code_block:
        errors.append(f'Unclosed code block started at line {code_block_start}')
    
    return {
        'code_blocks': code_blocks,
        'errors': errors,
        'total_blocks': len(code_blocks)
    }

# Check the file
filepath = '/Users/dannypan/PycharmProjects/xllm/lesson/5.模型执行器的原理.md'
result = check_markdown_file(filepath)

print(f'Total code blocks: {result["total_blocks"]}')
print(f'Errors: {len(result["errors"])}')
if result['errors']:
    print('\nErrors found:')
    for error in result['errors']:
        print(f'  - {error}')
else:
    print('\nNo errors found!')
    
print('\nCode blocks:')
for block in result['code_blocks']:
    print(f'  Block {block["type"]}: lines {block["start"]} to {block["end"]}')
