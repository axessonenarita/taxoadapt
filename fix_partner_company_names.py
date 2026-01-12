#!/usr/bin/env python3
"""
「導入パートナー」がcompany_nameになっているファイルを修正するスクリプト
基本情報セクションから実際の会社名を抽出して修正
"""
import os
import re
import glob
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def extract_actual_company_name_from_content(content):
    """Markdownファイルから実際の会社名を抽出（導入パートナー以外）"""
    # パターン1: 基本情報セクション内の最初の###見出し（導入パートナー以外）
    pattern1 = r'##\s+基本情報.*?###\s+([^#\n]+)'
    match1 = re.search(pattern1, content, re.DOTALL)
    if match1:
        company_name = match1.group(1).strip()
        company_name = company_name.replace('様', '').strip()
        
        # 「導入パートナー」で始まる場合はスキップ
        if company_name.startswith('導入パートナー'):
            # 次の###見出しを探す
            remaining = content[match1.end():]
            pattern_next = r'###\s+([^#\n]+)'
            match_next = re.search(pattern_next, remaining)
            if match_next:
                next_name = match_next.group(1).strip()
                next_name = next_name.replace('様', '').strip()
                if not next_name.startswith('導入パートナー'):
                    return next_name
        else:
            return company_name
    
    # パターン2: 本文中から会社名を抽出（タイトルの直後など）
    # 「導入パートナー」で始まらない会社名を探す
    pattern2 = r'(?:^|\n)([^#\n]+(?:株式会社|有限会社|合資会社|合名会社|合同会社|一般社団法人|財団法人|協同組合|組合|市|町|村|都|道|府|県)[^\n]*)'
    matches = re.finditer(pattern2, content, re.MULTILINE)
    for match in matches:
        company_name = match.group(1).strip()
        company_name = company_name.replace('様', '').strip()
        if not company_name.startswith('導入パートナー') and len(company_name) < 100:
            return company_name
    
    return None

def fix_partner_company_name(file_path):
    """導入パートナーがcompany_nameになっているファイルを修正"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if not content.startswith('---'):
        return False
    
    # フロントマターの終わりを探す
    frontmatter_end = content.find('---\n', 3)
    if frontmatter_end == -1:
        return False
    
    frontmatter_text = content[:frontmatter_end + 4]
    body = content[frontmatter_end + 4:]
    
    # company_nameが「導入パートナー」で始まるかチェック
    company_name_match = re.search(r'^company_name:\s*(.+)$', frontmatter_text, re.MULTILINE)
    if not company_name_match:
        return False
    
    company_name_value = company_name_match.group(1).strip()
    if not company_name_value.startswith('導入パートナー'):
        return False
    
    # 実際の会社名を抽出
    actual_company_name = extract_actual_company_name_from_content(content)
    if not actual_company_name:
        print(f"  会社名を抽出できませんでした: {os.path.basename(file_path)}")
        return False
    
    # フロントマターを更新
    frontmatter_text = re.sub(
        r'^company_name:\s*.+$',
        f'company_name: {actual_company_name}',
        frontmatter_text,
        flags=re.MULTILINE
    )
    
    # ファイルに書き戻し
    new_content = frontmatter_text + body
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    return True

def main():
    """すべてのMarkdownファイルをチェック"""
    casestudy_dir = 'assets/casestudy'
    md_files = glob.glob(os.path.join(casestudy_dir, '*.md'))
    md_files = [f for f in md_files if not os.path.basename(f).startswith('INDEX') and not os.path.basename(f).startswith('README')]
    md_files.sort()
    
    fixed_count = 0
    for md_file in md_files:
        if fix_partner_company_name(md_file):
            print(f"修正: {os.path.basename(md_file)}")
            fixed_count += 1
    
    print(f"\n修正したファイル数: {fixed_count}")

if __name__ == '__main__':
    main()
