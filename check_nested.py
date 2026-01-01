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
                # Already in a code block, this would be a nested block
                nested_errors.append(f'Line {i}: Opening new code block while another is already open (started at line {code_block_start}, type: {code_block_type})')
    
    return nested_errors

# Check all markdown files in lesson directory
lesson_dir = '/Users/dannypan/PycharmProjects/xllm/lesson'
md_files = [f for f in os.listdir(lesson_dir) if f.endswith('.md')]

print('Checking for nested code blocks in lesson directory...\n')
for md_file in sorted(md_files):
    filepath = os.path.join(lesson_dir, md_file)
    errors = check_nested_blocks(filepath)
    if errors:
        print(f'{md_file}: Found {len(errors)} nested block error(s):')
        for error in errors:
            print(f'  - {error}')
    else:
        print(f'{md_file}: OK')
    print()
