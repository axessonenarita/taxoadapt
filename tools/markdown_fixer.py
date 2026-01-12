#!/usr/bin/env python3
"""
Markdownファイルの構造を修正するツール
"""
import os
import re
import glob
import sys
import io

from .markdown_utils import (
    extract_company_name_from_content,
    extract_company_name_from_markdown,
    fix_frontmatter_newlines,
    get_all_markdown_files,
    setup_utf8_output
)

setup_utf8_output()


def check_and_fix_markdown_structure(file_path):
    """Markdownファイルの構造をチェックし、問題があれば修正"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    issues = []
    fixed = False
    
    if not content.startswith('---'):
        return issues, fixed
    
    # フロントマターの終わりを探す
    frontmatter_end = content.find('---\n', 3)
    if frontmatter_end == -1:
        return issues, fixed
    
    frontmatter_end += 4
    frontmatter_text = content[:frontmatter_end]
    body = content[frontmatter_end:]
    
    # company_nameをチェック
    company_name_match = re.search(r'^company_name:\s*(.+)$', frontmatter_text, re.MULTILINE)
    if not company_name_match:
        issues.append("company_nameがない")
        # 会社名を抽出して追加
        company_name = extract_company_name_from_content(content)
        if company_name:
            # フロントマターに追加
            frontmatter_lines = frontmatter_text.split('\n')
            # ---の後に追加
            if len(frontmatter_lines) > 1:
                frontmatter_lines.insert(1, f'company_name: {company_name}')
                frontmatter_text = '\n'.join(frontmatter_lines)
                fixed = True
        else:
            issues.append("会社名を抽出できませんでした")
    else:
        company_name_value = company_name_match.group(1).strip()
        # company_nameの値が長すぎる（説明文になっている）場合は修正
        if len(company_name_value) > 50 or '。' in company_name_value or ('、' in company_name_value and len(company_name_value) > 30):
            issues.append(f"company_nameの値が不正（長すぎるまたは説明文）: {company_name_value[:50]}...")
            # 正しい会社名を抽出
            correct_company_name = extract_company_name_from_content(content)
            if correct_company_name:
                frontmatter_text = re.sub(
                    r'^company_name:\s*.+$',
                    f'company_name: {correct_company_name}',
                    frontmatter_text,
                    flags=re.MULTILINE
                )
                fixed = True
            else:
                issues.append("正しい会社名を抽出できませんでした")
    
    # タイトルをチェック
    title_match = re.search(r'^#\s+(.+)$', body, re.MULTILINE)
    if not title_match:
        issues.append("最初のタイトルがない")
        # ファイル名からタイトルを抽出
        basename = os.path.basename(file_path)
        title = re.sub(r'^\d+_', '', basename).replace('.md', '')
        # フロントマターの後にタイトルを追加
        body = f'# {title}\n\n' + body
        fixed = True
    
    # フロントマターの改行を修正
    if '---#' in frontmatter_text + body:
        content = frontmatter_text + body
        content = fix_frontmatter_newlines(content)
        frontmatter_end = content.find('---\n', 3) + 4
        frontmatter_text = content[:frontmatter_end]
        body = content[frontmatter_end:]
        fixed = True
    
    # 修正した場合はファイルに書き戻し
    if fixed:
        new_content = frontmatter_text + body
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
    
    return issues, fixed


def fix_partner_company_names(file_path):
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
    
    # 実際の会社名を抽出（基本情報セクションから）
    pattern1 = r'##\s+基本情報.*?###\s+([^#\n]+)'
    match1 = re.search(pattern1, content, re.DOTALL)
    if match1:
        company_name = match1.group(1).strip()
        company_name = company_name.replace('様', '').strip()
        
        # 「導入パートナー」で始まる場合は次の###見出しを探す
        if company_name.startswith('導入パートナー'):
            remaining = content[match1.end():]
            pattern_next = r'###\s+([^#\n]+)'
            match_next = re.search(pattern_next, remaining)
            if match_next:
                next_name = match_next.group(1).strip()
                next_name = next_name.replace('様', '').strip()
                if not next_name.startswith('導入パートナー'):
                    company_name = next_name
        else:
            # フロントマターを更新
            frontmatter_text = re.sub(
                r'^company_name:\s*.+$',
                f'company_name: {company_name}',
                frontmatter_text,
                flags=re.MULTILINE
            )
            
            # ファイルに書き戻し
            new_content = frontmatter_text + body
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            return True
    
    return False


def fix_all_markdown_files(directory='assets/casestudy', fix_partner=True, fix_structure=True):
    """すべてのMarkdownファイルを修正"""
    md_files = get_all_markdown_files(directory)
    
    fixed_count = 0
    issues_count = 0
    
    for md_file in md_files:
        filename = os.path.basename(md_file)
        
        # 導入パートナーの修正
        if fix_partner:
            if fix_partner_company_names(md_file):
                print(f"修正（導入パートナー）: {filename}")
                fixed_count += 1
        
        # 構造の修正
        if fix_structure:
            issues, fixed = check_and_fix_markdown_structure(md_file)
            if fixed:
                print(f"修正（構造）: {filename}")
                fixed_count += 1
            if issues:
                issues_count += 1
                print(f"問題: {filename}")
                for issue in issues:
                    print(f"  - {issue}")
    
    print(f"\n修正したファイル数: {fixed_count}")
    print(f"問題があるファイル数: {issues_count}")
    
    return fixed_count, issues_count
