#!/usr/bin/env python3
"""
Markdownファイルから会社名を抽出するスクリプト
"""
import os
import re
import glob
import json
import sys
import io

# Windowsのコンソール出力のエンコーディング問題を回避
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def extract_company_name_from_markdown(file_path):
    """Markdownファイルから会社名を抽出"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # YAMLフロントマターから抽出
    if content.startswith('---'):
        frontmatter_end = content.find('---\n', 3)
        if frontmatter_end != -1:
            frontmatter_text = content[:frontmatter_end + 4]
            company_name_match = re.search(r'^company_name:\s*(.+)$', frontmatter_text, re.MULTILINE)
            if company_name_match:
                return company_name_match.group(1).strip()
    
    return None

def main():
    """すべてのMarkdownファイルから会社名を抽出"""
    casestudy_dir = 'assets/casestudy'
    md_files = glob.glob(os.path.join(casestudy_dir, '*.md'))
    md_files = [f for f in md_files if not os.path.basename(f).startswith('INDEX') and not os.path.basename(f).startswith('README')]
    md_files.sort()
    
    companies = {}
    for md_file in md_files:
        company_name = extract_company_name_from_markdown(md_file)
        if company_name:
            filename = os.path.basename(md_file)
            companies[filename] = company_name
    
    # JSONファイルに保存
    output_file = 'company_names_to_research.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(companies, f, ensure_ascii=False, indent=2)
    
    print(f"抽出した会社数: {len(companies)}")
    print(f"結果を {output_file} に保存しました")
    
    # 一意の会社名を表示
    unique_companies = sorted(set(companies.values()))
    print(f"\n一意の会社数: {len(unique_companies)}")
    print("\n会社名一覧:")
    for company in unique_companies:
        print(f"  - {company}")

if __name__ == '__main__':
    main()
