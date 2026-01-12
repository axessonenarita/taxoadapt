#!/usr/bin/env python3
"""
Markdownファイルに業種と売上規模を追加するスクリプト
"""
import os
import re
import glob
import sys
import io

# Windowsのコンソール出力のエンコーディング問題を回避
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 会社情報のデータベース（手動で追加・更新）
COMPANY_INFO = {
    "東急不動産ホールディングス株式会社": {
        "industry": "不動産",
        "revenue_size": "1000億円以上"
    },
    "ぺんてる株式会社": {
        "industry": "製造業",
        "revenue_size": "100億～500億円"
    },
    "株式会社ニシヤマ": {
        "industry": "商社",
        "revenue_size": "100億～500億円"
    },
    "大阪府四條畷市": {
        "industry": "地方公共団体",
        "revenue_size": "100億円未満"  # 地方公共団体は売上規模の概念が異なるが、便宜上
    },
    # 以下、他の会社も追加していく
}

def add_company_info_to_markdown(file_path, company_name):
    """Markdownファイルに業種と売上規模を追加"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 既に情報がある場合はスキップ
    if 'company_industry:' in content and not content.split('company_industry:')[1].split('\n')[0].strip().startswith('#'):
        return False
    
    if not content.startswith('---'):
        return False
    
    # フロントマターの終わりを探す
    frontmatter_end = content.find('---\n', 3)
    if frontmatter_end == -1:
        return False
    
    frontmatter_end += 4
    frontmatter = content[:frontmatter_end]
    body = content[frontmatter_end:]
    
    # 会社情報を取得
    if company_name not in COMPANY_INFO:
        print(f"  情報が見つかりません: {company_name}")
        return False
    
    info = COMPANY_INFO[company_name]
    
    # フロントマターに追加
    frontmatter_lines = frontmatter.rstrip().split('\n')
    
    # company_nameの後に追加
    new_lines = []
    company_name_added = False
    industry_added = False
    revenue_added = False
    
    for line in frontmatter_lines:
        new_lines.append(line)
        
        # company_nameの後に業種と売上規模を追加
        if line.startswith('company_name:') and not company_name_added:
            new_lines.append(f'company_industry: {info["industry"]}')
            new_lines.append(f'company_revenue_size: {info["revenue_size"]}')
            company_name_added = True
            industry_added = True
            revenue_added = True
        
        # 既存のコメントアウトされた行を削除
        if line.strip().startswith('# company_industry:') or line.strip().startswith('# company_revenue_size:'):
            continue
    
    # コメントアウトされた行が残っている場合は削除
    final_lines = []
    for line in new_lines:
        if not (line.strip().startswith('# company_industry:') or line.strip().startswith('# company_revenue_size:')):
            final_lines.append(line)
    
    new_frontmatter = '\n'.join(final_lines)
    
    # ファイルに書き戻し
    new_content = new_frontmatter + body
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    return True

def extract_company_name_from_markdown(file_path):
    """Markdownファイルから会社名を抽出"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if content.startswith('---'):
        frontmatter_end = content.find('---\n', 3)
        if frontmatter_end != -1:
            frontmatter_text = content[:frontmatter_end + 4]
            company_name_match = re.search(r'^company_name:\s*(.+)$', frontmatter_text, re.MULTILINE)
            if company_name_match:
                return company_name_match.group(1).strip()
    
    return None

def main():
    """すべてのMarkdownファイルに会社情報を追加"""
    casestudy_dir = 'assets/casestudy'
    md_files = glob.glob(os.path.join(casestudy_dir, '*.md'))
    md_files = [f for f in md_files if not os.path.basename(f).startswith('INDEX') and not os.path.basename(f).startswith('README')]
    md_files.sort()
    
    updated_count = 0
    not_found_count = 0
    
    for md_file in md_files:
        company_name = extract_company_name_from_markdown(md_file)
        if company_name:
            filename = os.path.basename(md_file)
            if add_company_info_to_markdown(md_file, company_name):
                print(f"更新: {filename}")
                updated_count += 1
            elif company_name not in COMPANY_INFO:
                print(f"情報なし: {filename} - {company_name}")
                not_found_count += 1
    
    print(f"\n更新したファイル数: {updated_count}")
    print(f"情報が見つからなかったファイル数: {not_found_count}")

if __name__ == '__main__':
    main()
