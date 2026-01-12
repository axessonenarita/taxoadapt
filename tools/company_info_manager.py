#!/usr/bin/env python3
"""
会社情報管理ツール
"""
import os
import json
import re
import glob
from tqdm import tqdm
import sys
import io

from .markdown_utils import (
    extract_company_name_from_markdown,
    extract_company_name_from_content,
    extract_metadata_from_markdown,
    get_all_markdown_files,
    setup_utf8_output
)

setup_utf8_output()


# 会社情報のデータベース（手動で追加・更新）
COMPANY_INFO_DB = {
    "東急不動産ホールディングス株式会社": {"industry": "不動産", "revenue_size": "1000億円以上"},
    "ぺんてる株式会社": {"industry": "製造業", "revenue_size": "100億～500億円"},
    "株式会社ニシヤマ": {"industry": "商社", "revenue_size": "100億～500億円"},
    "大阪府四條畷市": {"industry": "地方公共団体", "revenue_size": "100億円未満"},
    # その他の会社情報は add_company_info_batch.py から移行
}


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


def generate_company_info_json(output_file='datasets/casestudy/company_info.json'):
    """すべてのMarkdownファイルから会社情報を抽出してJSONを生成"""
    data_dir = os.path.dirname(output_file)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    md_files = get_all_markdown_files()
    company_info_dict = {}
    
    for md_file in tqdm(md_files, desc="Extracting company info"):
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        company_name = extract_company_name_from_markdown(md_file)
        if company_name:
            company_info = extract_company_info_from_markdown(content, company_name)
            if company_info:
                company_info_dict[company_name] = company_info
    
    # JSONファイルに保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(company_info_dict, f, ensure_ascii=False, indent=2)
    
    print(f"\n生成完了: {output_file}")
    print(f"抽出した会社数: {len(company_info_dict)}")
    
    return company_info_dict


def add_company_info_to_markdown(file_path, company_name, company_info_db=None):
    """Markdownファイルに業種と売上規模を追加"""
    if company_info_db is None:
        company_info_db = COMPANY_INFO_DB
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if not content.startswith('---'):
        return False
    
    # フロントマターの終わりを探す
    frontmatter_end = content.find('---\n', 3)
    if frontmatter_end == -1:
        return False
    
    frontmatter_end += 4
    frontmatter = content[:frontmatter_end]
    body = content[frontmatter_end:]
    
    # 既に情報がある場合はスキップ（コメントアウトされていない場合）
    if 'company_industry:' in content:
        industry_line = [line for line in content.split('\n') if 'company_industry:' in line][0]
        if not industry_line.strip().startswith('#'):
            return False
    
    # 会社情報を取得（部分一致も試す）
    info = None
    if company_name in company_info_db:
        info = company_info_db[company_name]
    else:
        # 部分一致で検索
        for key, value in company_info_db.items():
            if company_name in key or key in company_name:
                info = value
                break
    
    if not info:
        return False
    
    # フロントマターを更新
    frontmatter_lines = frontmatter.rstrip().split('\n')
    new_lines = []
    
    for line in frontmatter_lines:
        # コメントアウトされた業種・売上規模の行を削除
        if line.strip().startswith('# company_industry:') or line.strip().startswith('# company_revenue_size:'):
            continue
        
        new_lines.append(line)
        
        # company_nameの後に業種と売上規模を追加
        if line.startswith('company_name:'):
            new_lines.append(f'company_industry: {info["industry"]}')
            new_lines.append(f'company_revenue_size: {info["revenue_size"]}')
    
    new_frontmatter = '\n'.join(new_lines)
    
    # ファイルに書き戻し
    new_content = new_frontmatter + body
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    return True
