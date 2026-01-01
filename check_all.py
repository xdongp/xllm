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

# Check all markdown files in lesson directory
lesson_dir = '/Users/dannypan/PycharmProjects/xllm/lesson'
md_files = [f for f in os.listdir(lesson_dir) if f.endswith('.md')]

print('Checking all markdown files in lesson directory...\n')
all_ok = True
for md_file in sorted(md_files):
    filepath = os.path.join(lesson_dir, md_file)
    result = check_markdown_file(filepath)
    if result['errors']:
        all_ok = False
        print(f'{md_file}: Found {len(result["errors"])} error(s)')
        for error in result['errors']:
            print(f'  - {error}')
    else:
        print(f'{md_file}: OK ({result["total_blocks"]} code blocks)')

print()
if all_ok:
    print('✓ All markdown files are OK!')
else:
    print('✗ Some files have errors!')
