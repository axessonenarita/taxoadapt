#!/usr/bin/env python3
"""
Markdownファイルの構造をチェックし、問題があるファイルを修正するスクリプト
"""
import os
import re
import glob
import sys
import io

# Windowsのコンソール出力のエンコーディング問題を回避
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def extract_company_name_from_content(content):
    """Markdownファイルから会社名を抽出"""
    # パターン1: 「### 会社名」形式（基本情報セクション内）- 最も確実
    # 基本情報セクション内の最初の###見出しを取得
    pattern1 = r'##\s+基本情報.*?###\s+([^#\n]+)'
    match1 = re.search(pattern1, content, re.DOTALL)
    if match1:
        company_name = match1.group(1).strip()
        company_name = company_name.replace('様', '').strip()
        # 長すぎる場合は最初の会社名部分だけを抽出
        if len(company_name) > 50:
            # 「（以下、」の前まで
            match_short = re.search(r'^([^（]+)', company_name)
            if match_short:
                company_name = match_short.group(1).strip()
        # 空でない場合のみ返す
        if company_name:
            return company_name
    
    # パターン2: 「### 会社名」形式（基本情報セクション外）
    # 会社名のパターン: 株式会社、有限会社、合資会社、合名会社、合同会社、一般社団法人、財団法人、協同組合、組合、市、町、村、都、道、府、県
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
    
    # パターン3: 「会社名様」形式（タイトルの直後）
    pattern3 = r'^#\s+[^\n]+\n+\n+([^#\n]+' + company_pattern + r'[^\n]*様?)'
    match3 = re.search(pattern3, content, re.MULTILINE)
    if match3:
        company_name = match3.group(1).strip()
        company_name = company_name.replace('様', '').strip()
        if len(company_name) > 50:
            match_short = re.search(r'^([^（]+)', company_name)
            if match_short:
                company_name = match_short.group(1).strip()
        return company_name
    
    return None

def extract_title_from_filename(file_path):
    """ファイル名からタイトルを抽出"""
    basename = os.path.basename(file_path)
    title = re.sub(r'^\d+_', '', basename).replace('.md', '')
    return title

def check_and_fix_markdown(file_path):
    """Markdownファイルの構造をチェックし、問題があれば修正"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    issues = []
    fixed = False
    
    # フロントマターがあるかチェック
    if not content.startswith('---'):
        issues.append("フロントマターがない")
        return issues, False
    
    # フロントマターの終わりを探す
    frontmatter_end = content.find('---\n', 3)
    if frontmatter_end == -1:
        issues.append("フロントマターが正しく閉じられていない")
        return issues, False
    
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
            # 会社名が見つからない場合は、ファイル名から推測を試みる
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
    if not body.strip().startswith('#'):
        issues.append("最初のタイトルがない")
        # ファイル名からタイトルを抽出して追加
        title = extract_title_from_filename(file_path)
        body = f'# {title}\n\n{body}'
        fixed = True
    
    # 修正があった場合はファイルに書き戻し
    if fixed:
        new_content = frontmatter_text + body
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
    
    return issues, fixed

def main():
    """すべてのMarkdownファイルをチェック"""
    casestudy_dir = 'assets/casestudy'
    md_files = glob.glob(os.path.join(casestudy_dir, '*.md'))
    md_files = [f for f in md_files if not os.path.basename(f).startswith('INDEX')]
    md_files.sort()
    
    problem_files = []
    fixed_count = 0
    
    for md_file in md_files:
        issues, fixed = check_and_fix_markdown(md_file)
        if issues:
            problem_files.append((os.path.basename(md_file), issues))
        if fixed:
            fixed_count += 1
    
    # 問題があるファイルを表示
    if problem_files:
        print("問題があるファイル:")
        for filename, issues in problem_files:
            print(f"  {filename}:")
            for issue in issues:
                print(f"    - {issue}")
        print()
    
    print(f"修正したファイル数: {fixed_count}")
    print(f"問題があるファイル数: {len(problem_files)}")

if __name__ == '__main__':
    main()
