#!/usr/bin/env python3
"""
既存のフロントマターがあるMarkdownファイルにタイトルを追加するスクリプト
"""
import os
import re
import glob
import sys
import io

# Windowsのコンソール出力のエンコーディング問題を回避
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def extract_title_from_content(content):
    """Markdownファイルからタイトルを抽出（2つ目の#見出し、またはファイル名から）"""
    # パターン: 導入事例\n\n# タイトル
    title_pattern = r'導入事例\n+\n+#\s+([^\n]+)'
    match = re.search(title_pattern, content)
    if match:
        return match.group(1).strip()
    
    # ファイル名から抽出
    return None

def fix_markdown_title(file_path):
    """Markdownファイルにタイトルを追加"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 既にフロントマターがある場合のみ処理
    if not content.startswith('---'):
        return False
    
    # フロントマターの終わりを探す
    frontmatter_end = content.find('---\n', 3)  # 2つ目の---
    if frontmatter_end == -1:
        return False
    
    frontmatter_end += 4  # '---\n'の長さ
    frontmatter = content[:frontmatter_end]
    body = content[frontmatter_end:]
    
    # 本文の最初がタイトルでない場合、タイトルを追加
    if not body.strip().startswith('#'):
        # ファイル名からタイトルを抽出
        basename = os.path.basename(file_path)
        title = re.sub(r'^\d+_', '', basename).replace('.md', '')
        
        # 本文の先頭にタイトルを追加
        body = f'# {title}\n\n{body}'
        
        # ファイルに書き戻し
        new_content = frontmatter + body
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"Fixed: {os.path.basename(file_path)}")
        return True
    
    return False

def main():
    """すべてのMarkdownファイルを修正"""
    casestudy_dir = 'assets/casestudy'
    md_files = glob.glob(os.path.join(casestudy_dir, '*.md'))
    md_files = [f for f in md_files if not os.path.basename(f).startswith('INDEX')]
    md_files.sort()
    
    fixed_count = 0
    for md_file in md_files:
        if fix_markdown_title(md_file):
            fixed_count += 1
    
    print(f"\nTotal fixed: {fixed_count} files")

if __name__ == '__main__':
    main()
