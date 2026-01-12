#!/usr/bin/env python3
"""
Markdownファイルから会社情報を抽出してcompany_info.jsonを生成するスクリプト
"""
import os
import json
import re
import glob
from tqdm import tqdm
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def extract_company_name(content):
    """Markdownファイルから会社名を抽出"""
    # YAMLフロントマターから抽出
    if content.startswith('---'):
        frontmatter_end = content.find('---\n', 3)
        if frontmatter_end != -1:
            frontmatter_text = content[:frontmatter_end + 4]
            company_name_match = re.search(r'^company_name:\s*(.+)$', frontmatter_text, re.MULTILINE)
            if company_name_match:
                return company_name_match.group(1).strip()
    return None

def extract_metadata_from_markdown(content):
    """MarkdownファイルのYAMLフロントマターからメタデータを抽出"""
    frontmatter_pattern = r'^---\s*\n(.*?)\n---\s*\n'
    frontmatter_match = re.search(frontmatter_pattern, content, re.DOTALL | re.MULTILINE)
    
    if not frontmatter_match:
        return None
    
    frontmatter_text = frontmatter_match.group(1)
    metadata = {}
    
    for line in frontmatter_text.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            value = value.strip('"\'')
            metadata[key] = value
    
    return metadata if metadata else None

def extract_company_info_from_markdown(content, company_name):
    """Markdownファイルから会社情報を抽出"""
    if not company_name:
        return None
    
    # YAMLフロントマターから抽出
    metadata = extract_metadata_from_markdown(content)
    if metadata:
        company_info = {}
        if 'company_industry' in metadata or 'industry' in metadata:
            company_info['industry'] = metadata.get('company_industry') or metadata.get('industry')
        if 'company_revenue_size' in metadata or 'revenue_size' in metadata:
            company_info['revenue_size'] = metadata.get('company_revenue_size') or metadata.get('revenue_size')
        
        if company_info.get('industry') or company_info.get('revenue_size'):
            return company_info
    
    return None

def main():
    """すべてのMarkdownファイルから会社情報を抽出してJSONを生成"""
    casestudy_dir = 'assets/casestudy'
    data_dir = 'datasets/casestudy'
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    md_files = glob.glob(os.path.join(casestudy_dir, '*.md'))
    md_files = [f for f in md_files if not os.path.basename(f).startswith('INDEX') and not os.path.basename(f).startswith('README')]
    md_files.sort()
    
    company_info_dict = {}
    
    for md_file in tqdm(md_files, desc="Extracting company info"):
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        company_name = extract_company_name(content)
        if company_name:
            company_info = extract_company_info_from_markdown(content, company_name)
            if company_info:
                company_info_dict[company_name] = company_info
    
    # JSONファイルに保存
    output_file = os.path.join(data_dir, 'company_info.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(company_info_dict, f, ensure_ascii=False, indent=2)
    
    print(f"\n生成完了: {output_file}")
    print(f"抽出した会社数: {len(company_info_dict)}")
    
    # 統計情報を表示
    industries = {}
    revenue_sizes = {}
    for company, info in company_info_dict.items():
        if info.get('industry'):
            industries[info['industry']] = industries.get(info['industry'], 0) + 1
        if info.get('revenue_size'):
            revenue_sizes[info['revenue_size']] = revenue_sizes.get(info['revenue_size'], 0) + 1
    
    print(f"\n業種別の分布:")
    for industry, count in sorted(industries.items(), key=lambda x: x[1], reverse=True):
        print(f"  {industry}: {count}")
    
    print(f"\n売上規模別の分布:")
    for size, count in sorted(revenue_sizes.items(), key=lambda x: x[1], reverse=True):
        print(f"  {size}: {count}")

if __name__ == '__main__':
    main()
