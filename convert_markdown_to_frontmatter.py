#!/usr/bin/env python3
"""
MarkdownファイルをYAMLフロントマター形式に変換するスクリプト
"""
import os
import re
import glob

def extract_company_name_from_content(content):
    """Markdownファイルから会社名を抽出"""
    # パターン1: 「### 会社名」形式（基本情報セクション内）
    pattern1 = r'###\s+([^#\n]+(?:株式会社|有限会社|合資会社|合名会社|合同会社|一般社団法人|財団法人|協同組合|組合)[^\n]*)'
    match1 = re.search(pattern1, content)
    if match1:
        company_name = match1.group(1).strip()
        company_name = company_name.replace('様', '').strip()
        return company_name
    
    # パターン2: 「会社名様」形式（タイトルの直後）
    pattern2 = r'^#\s+[^\n]+\n+\n+([^#\n]+(?:株式会社|有限会社|合資会社|合名会社|合同会社|一般社団法人|財団法人|協同組合|組合)[^\n]*様?)'
    match2 = re.search(pattern2, content, re.MULTILINE)
    if match2:
        company_name = match2.group(1).strip()
        company_name = company_name.replace('様', '').strip()
        return company_name
    
    return None

def extract_url_from_content(content):
    """MarkdownファイルからURLを抽出"""
    url_pattern = r'\*\*URL\*\*:\s*(https?://[^\s\n]+)'
    match = re.search(url_pattern, content)
    if match:
        return match.group(1)
    return None

def extract_title_from_content(content):
    """Markdownファイルからタイトルを抽出（最初の#見出し）"""
    title_pattern = r'^#\s+(.+)$'
    match = re.search(title_pattern, content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None

def convert_markdown_to_frontmatter(file_path):
    """MarkdownファイルをYAMLフロントマター形式に変換"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 既にフロントマターがある場合はスキップ
    if content.startswith('---'):
        print(f"Skipping {file_path} (already has frontmatter)")
        return False
    
    # 情報を抽出
    title = extract_title_from_content(content)
    url = extract_url_from_content(content)
    company_name = extract_company_name_from_content(content)
    
    # 既存の先頭部分を削除（最初のタイトル、URL、区切り線、導入事例、タイトル、会社名まで）
    # パターン: # タイトル\n\n**URL**: ...\n\n---\n\n導入事例\n\n# タイトル\n\n会社名様\n\n
    # タイトル（2つ目）は残す必要がある
    # まず、2つ目のタイトルを抽出
    title_pattern = r'導入事例\n+\n+#\s+([^\n]+)'
    title_match = re.search(title_pattern, content)
    second_title = title_match.group(1).strip() if title_match else title
    
    # パターンで削除
    pattern_to_remove = r'^#\s+[^\n]+\n+\n+\*\*URL\*\*:\s*[^\n]+\n+\n+---\n+\n+導入事例\n+\n+#\s+[^\n]+\n+\n+[^\n]+様?\n+\n+'
    new_content = re.sub(pattern_to_remove, '', content, flags=re.MULTILINE)
    
    # もしパターンがマッチしなかった場合、より柔軟なパターンを試す
    if new_content == content:
        # 最初のタイトル、URL、区切り線、導入事例、会社名まで削除（タイトルは残す）
        pattern_to_remove2 = r'^#\s+[^\n]+\n+\n+\*\*URL\*\*:\s*[^\n]+\n+\n+---\n+\n+導入事例\n+\n+#\s+[^\n]+\n+\n+[^\n]+様?\n+\n+'
        new_content = re.sub(pattern_to_remove2, '', content, flags=re.MULTILINE)
    
    # タイトルが削除されてしまった場合は、抽出したタイトルを先頭に追加
    if second_title and not new_content.strip().startswith('#'):
        new_content = f'# {second_title}\n\n{new_content}'
    
    # YAMLフロントマターを構築
    frontmatter_lines = ['---']
    
    if company_name:
        frontmatter_lines.append(f'company_name: {company_name}')
    
    # 業種と売上規模は後で手動入力するため、コメントアウト
    frontmatter_lines.append('# company_industry: # 業種を入力してください（例: 不動産、製造業、金融、小売、ITなど）')
    frontmatter_lines.append('# company_revenue_size: # 売上規模を入力してください（1000億円以上、500億～1000億円、100億～500億円、100億円未満）')
    
    if url:
        frontmatter_lines.append(f'url: {url}')
    
    frontmatter_lines.append('---')
    frontmatter_lines.append('')
    
    # 新しい内容を構築
    new_content_with_frontmatter = '\n'.join(frontmatter_lines) + '\n' + new_content
    
    # ファイルに書き戻し
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content_with_frontmatter)
    
    print(f"Converted: {os.path.basename(file_path)}")
    return True

def main():
    """すべてのMarkdownファイルを変換"""
    import sys
    import io
    
    # Windowsのコンソール出力のエンコーディング問題を回避
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    casestudy_dir = 'assets/casestudy'
    md_files = glob.glob(os.path.join(casestudy_dir, '*.md'))
    md_files = [f for f in md_files if not os.path.basename(f).startswith('INDEX')]
    md_files.sort()
    
    converted_count = 0
    for md_file in md_files:
        if convert_markdown_to_frontmatter(md_file):
            converted_count += 1
    
    print(f"\nTotal converted: {converted_count} files")

if __name__ == '__main__':
    main()
