#!/usr/bin/env python3
"""
フロントマターの改行を修正するスクリプト
"""
import os
import glob
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def fix_frontmatter_newlines(file_path):
    """フロントマターの改行を修正"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if not content.startswith('---'):
        return False
    
    # ---# のパターンを修正
    if '---#' in content:
        content = content.replace('---#', '---\n\n#')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    
    return False

def main():
    """すべてのMarkdownファイルを修正"""
    casestudy_dir = 'assets/casestudy'
    md_files = glob.glob(os.path.join(casestudy_dir, '*.md'))
    md_files = [f for f in md_files if not os.path.basename(f).startswith('INDEX') and not os.path.basename(f).startswith('README')]
    md_files.sort()
    
    fixed_count = 0
    for md_file in md_files:
        if fix_frontmatter_newlines(md_file):
            fixed_count += 1
            print(f"修正: {os.path.basename(md_file)}")
    
    print(f"\n修正したファイル数: {fixed_count}")

if __name__ == '__main__':
    main()
