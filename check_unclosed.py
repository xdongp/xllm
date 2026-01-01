#!/usr/bin/env python3
import os

def check_nested_blocks(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    in_code_block = False
    code_block_start = None
    code_block_type = None
    nested_errors = []
    
    for i, line in enumerate(lines, 1):
        if line.strip().startswith('```'):
            if not in_code_block:
                # Opening a code block
                in_code_block = True
                code_block_start = i
                code_block_type = line.strip()[3:].strip() or 'code'
            else:
                # This should be a closing marker
                in_code_block = False
                code_block_start = None
                code_block_type = None
    
    if in_code_block:
        nested_errors.append(f'Unclosed code block started at line {code_block_start}')
    
    return nested_errors

# Check only the 5th file
filepath = '/Users/dannypan/PycharmProjects/xllm/lesson/5.模型执行器的原理.md'
errors = check_nested_blocks(filepath)

print(f'Checking 5.模型执行器的原理.md for unclosed code blocks...\n')
if errors:
    print(f'Found {len(errors)} error(s):')
    for error in errors:
        print(f'  - {error}')
else:
    print('OK - All code blocks are properly closed!')
