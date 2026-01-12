#!/usr/bin/env python3
"""
Markdownファイル処理ユーティリティ
"""
import os
import re
import glob
import sys
import io

# Windowsのコンソール出力のエンコーディング問題を回避（必要に応じて）
def setup_utf8_output():
    """WindowsでのUTF-8出力を設定"""
    if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass  # 既に設定されているか、設定できない場合はスキップ


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


def extract_company_name_from_content(content):
    """Markdownファイルの内容から会社名を抽出"""
    # パターン1: 「### 会社名」形式（基本情報セクション内）- 最も確実
    pattern1 = r'##\s+基本情報.*?###\s+([^#\n]+)'
    match1 = re.search(pattern1, content, re.DOTALL)
    if match1:
        company_name = match1.group(1).strip()
        company_name = company_name.replace('様', '').strip()
        # 長すぎる場合は最初の会社名部分だけを抽出
        if len(company_name) > 50:
            match_short = re.search(r'^([^（]+)', company_name)
            if match_short:
                company_name = match_short.group(1).strip()
        if company_name:
            return company_name
    
    # パターン2: 「### 会社名」形式（基本情報セクション外）
    company_pattern = r'(?:株式会社|有限会社|合資会社|合名会社|合同会社|一般社団法人|財団法人|協同組合|組合|市|町|村|都|道|府|県|PT\.|Ltd\.|Inc\.)'
    pattern2 = r'###\s+([^#\n]+' + company_pattern + r'[^\n]*)'
    match2 = re.search(pattern2, content)
    if match2:
        company_name = match2.group(1).strip()
        company_name = company_name.replace('様', '').strip()
        if len(company_name) > 50:
            match_short = re.search(r'^([^（]+)', company_name)
            if match_short:
                company_name = match_short.group(1).strip()
        return company_name
    
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


def fix_frontmatter_newlines(content):
    """フロントマターの改行を修正（---# を ---\n\n# に）"""
    if '---#' in content:
        content = content.replace('---#', '---\n\n#')
    return content


def get_all_markdown_files(directory='assets/casestudy', exclude_patterns=None):
    """指定ディレクトリ内のすべてのMarkdownファイルを取得"""
    if exclude_patterns is None:
        exclude_patterns = ['INDEX', 'README']
    
    md_files = glob.glob(os.path.join(directory, '*.md'))
    md_files = [f for f in md_files 
                 if not any(os.path.basename(f).startswith(pattern) for pattern in exclude_patterns)]
    md_files.sort()
    return md_files
