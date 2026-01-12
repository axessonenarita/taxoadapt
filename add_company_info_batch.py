#!/usr/bin/env python3
"""
Markdownファイルに業種と売上規模を一括追加するスクリプト
会社情報は手動で調査してCOMPANY_INFOに追加してください
"""
import os
import re
import glob
import sys
import io

# Windowsのコンソール出力のエンコーディング問題を回避
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 会社情報のデータベース（Web検索で調査した結果をここに追加）
# フォーマット: "会社名": {"industry": "業種", "revenue_size": "売上規模"}
COMPANY_INFO = {
    # 主要な会社（調査済み）
    "東急不動産ホールディングス株式会社": {"industry": "不動産", "revenue_size": "1000億円以上"},
    "ぺんてる株式会社": {"industry": "製造業", "revenue_size": "100億～500億円"},
    "株式会社ニシヤマ": {"industry": "商社", "revenue_size": "100億～500億円"},
    "大阪府四條畷市": {"industry": "地方公共団体", "revenue_size": "100億円未満"},
    "ヤマトクレジットファイナンス株式会社": {"industry": "金融", "revenue_size": "100億～500億円"},
    "第一三共株式会社": {"industry": "製造業", "revenue_size": "1000億円以上"},
    "ロート製薬株式会社": {"industry": "製造業", "revenue_size": "1000億円以上"},
    "NTN株式会社": {"industry": "製造業", "revenue_size": "1000億円以上"},
    "三菱ＵＦＪニコス株式会社": {"industry": "金融", "revenue_size": "1000億円以上"},
    "九州旅客鉄道株式会社": {"industry": "運輸", "revenue_size": "1000億円以上"},
    "三井住友トラスト・アセットマネジメント株式会社": {"industry": "金融", "revenue_size": "1000億円以上"},
    "エクシオグループ株式会社": {"industry": "IT", "revenue_size": "1000億円以上"},
    "東レ・メディカル株式会社": {"industry": "製造業", "revenue_size": "500億～1000億円"},
    "光洋機械産業株式会社": {"industry": "製造業", "revenue_size": "100億～500億円"},
    "帝都自動車交通株式会社": {"industry": "運輸", "revenue_size": "100億～500億円"},
    "アポクリート株式会社": {"industry": "小売", "revenue_size": "100億～500億円"},
    "大同火災海上保険株式会社": {"industry": "保険", "revenue_size": "1000億円以上"},
    "石油資源開発株式会社": {"industry": "エネルギー", "revenue_size": "1000億円以上"},
    "日立造船マリンエンジン株式会社": {"industry": "製造業", "revenue_size": "500億～1000億円"},
    "三洋化成工業株式会社": {"industry": "製造業", "revenue_size": "100億～500億円"},
    "西華産業株式会社": {"industry": "商社", "revenue_size": "100億～500億円"},
    "太陽化学株式会社": {"industry": "製造業", "revenue_size": "100億～500億円"},
    "株式会社資生堂": {"industry": "製造業", "revenue_size": "1000億円以上"},
    "株式会社三菱総合研究所": {"industry": "IT", "revenue_size": "1000億円以上"},
    "株式会社日立マネジメントパートナー": {"industry": "サービス", "revenue_size": "1000億円以上"},
    "株式会社アンデルセンサービス": {"industry": "サービス", "revenue_size": "100億～500億円"},
    "株式会社オプテージ": {"industry": "IT", "revenue_size": "1000億円以上"},
    "株式会社明電舎": {"industry": "製造業", "revenue_size": "1000億円以上"},
    "株式会社理経": {"industry": "IT", "revenue_size": "100億～500億円"},
    "森ビル株式会社": {"industry": "不動産", "revenue_size": "1000億円以上"},
    "石川県": {"industry": "地方公共団体", "revenue_size": "1000億円以上"},
    "ＳＯＭＰＯホールディングス株式会社": {"industry": "保険", "revenue_size": "1000億円以上"},
    "伊藤忠商事株式会社": {"industry": "商社", "revenue_size": "1000億円以上"},
    "東北電力株式会社": {"industry": "エネルギー", "revenue_size": "1000億円以上"},
    "東日本電信電話株式会社": {"industry": "通信", "revenue_size": "1000億円以上"},
    "全日本空輸株式会社": {"industry": "運輸", "revenue_size": "1000億円以上"},
    "西武鉄道株式会社": {"industry": "運輸", "revenue_size": "1000億円以上"},
    "中部国際空港株式会社": {"industry": "運輸", "revenue_size": "100億～500億円"},
    "学校法人千葉工業大学": {"industry": "教育", "revenue_size": "100億円未満"},
    "ヤフー株式会社": {"industry": "IT", "revenue_size": "1000億円以上"},
    "ブラザー工業株式会社": {"industry": "製造業", "revenue_size": "1000億円以上"},
    "トヨタファイナンス株式会社": {"industry": "金融", "revenue_size": "1000億円以上"},
    "オリックス生命保険株式会社": {"industry": "保険", "revenue_size": "1000億円以上"},
    "赤城乳業株式会社": {"industry": "製造業", "revenue_size": "100億～500億円"},
    "大日本除虫菊株式会社": {"industry": "製造業", "revenue_size": "100億～500億円"},
    "三菱マテリアル株式会社": {"industry": "製造業", "revenue_size": "1000億円以上"},
    "三菱電機ビルテクノサービス株式会社": {"industry": "サービス", "revenue_size": "500億～1000億円"},
    "日本特殊陶業株式会社": {"industry": "製造業", "revenue_size": "1000億円以上"},
    "日立造船マリンエンジン株式会社": {"industry": "製造業", "revenue_size": "500億～1000億円"},
    "日精機工株式会社": {"industry": "製造業", "revenue_size": "100億～500億円"},
    "東ソー情報システム株式会社": {"industry": "IT", "revenue_size": "100億～500億円"},
    "東洋鋼鈑株式会社": {"industry": "製造業", "revenue_size": "1000億円以上"},
    "川崎重工業株式会社": {"industry": "製造業", "revenue_size": "1000億円以上"},
    "帝人エンジニアリング株式会社": {"industry": "製造業", "revenue_size": "500億～1000億円"},
    "兼松株式会社": {"industry": "商社", "revenue_size": "1000億円以上"},
    "地盤ネットホールディングス株式会社": {"industry": "建設", "revenue_size": "100億～500億円"},
    "北海道エネルギー株式会社": {"industry": "エネルギー", "revenue_size": "100億～500億円"},
    "トーソー株式会社": {"industry": "製造業", "revenue_size": "100億～500億円"},
    "トッパン・フォームズ株式会社": {"industry": "製造業", "revenue_size": "1000億円以上"},
    "ジャパンシステム株式会社": {"industry": "IT", "revenue_size": "100億～500億円"},
    "エイチアールワン株式会社": {"industry": "IT", "revenue_size": "100億～500億円"},
    "イオンアイビス株式会社": {"industry": "IT", "revenue_size": "1000億円以上"},
    "アットホーム株式会社": {"industry": "不動産", "revenue_size": "1000億円以上"},
    "アルプス システム インテグレーション株式会社": {"industry": "IT", "revenue_size": "1000億円以上"},
    "さくら情報システム株式会社": {"industry": "IT", "revenue_size": "100億～500億円"},
    "I＆H株式会社": {"industry": "IT", "revenue_size": "100億～500億円"},
    "NTTコミュニケーションズ株式会社": {"industry": "通信", "revenue_size": "1000億円以上"},
    "株式会社NTTデータMSE": {"industry": "IT", "revenue_size": "100億～500億円"},
    "株式会社NTTファシリティーズ": {"industry": "サービス", "revenue_size": "1000億円以上"},
    "株式会社b-ex": {"industry": "サービス", "revenue_size": "100億～500億円"},
    "株式会社システック": {"industry": "IT", "revenue_size": "100億～500億円"},
    "株式会社トヨテック": {"industry": "IT", "revenue_size": "100億～500億円"},
    "株式会社フォーバルテレコム": {"industry": "通信", "revenue_size": "100億～500億円"},
    "株式会社 日立ICTビジネスサービス": {"industry": "IT", "revenue_size": "1000億円以上"},
    "株式会社ATビジネス": {"industry": "サービス", "revenue_size": "100億～500億円"},
    # 追加の会社情報
    "三菱総合研究所": {"industry": "IT", "revenue_size": "1000億円以上"},
    "日立マネジメントパートナー": {"industry": "サービス", "revenue_size": "1000億円以上"},
    "伊藤忠商事": {"industry": "商社", "revenue_size": "1000億円以上"},
    "タイ・エアアジア（Thai AirAsia）": {"industry": "運輸", "revenue_size": "100億～500億円"},
    "佳能光学設備": {"industry": "製造業", "revenue_size": "100億～500億円"},
    "東電化（中国）投資有限公司": {"industry": "製造業", "revenue_size": "100億～500億円"},
    "NECネクサソリューションズ株式会社": {"industry": "IT", "revenue_size": "1000億円以上"},
    "ヤフー株式会社Yahoo Japan Corporation": {"industry": "IT", "revenue_size": "1000億円以上"},
    "株式会社b-ex(旧:株式会社ビューティーエクスペリエンス)": {"industry": "サービス", "revenue_size": "100億～500億円"},
    "東ソー情報システム株式会社（南陽事業所）": {"industry": "IT", "revenue_size": "100億～500億円"},
    "川崎重工業株式会社　プラント・環境カンパニー": {"industry": "製造業", "revenue_size": "1000億円以上"},
    "東日本電信電話株式会社（本社）": {"industry": "通信", "revenue_size": "1000億円以上"},
    # 導入パートナー
    "導入パートナー　NECネクサソリューションズ株式会社": {"industry": "IT", "revenue_size": "1000億円以上"},
    "導入パートナー　エクシオ・デジタルソリューションズ株式会社": {"industry": "IT", "revenue_size": "1000億円以上"},
    "導入パートナー　パーソルプロセス&テクノロジー株式会社": {"industry": "IT", "revenue_size": "1000億円以上"},
    "導入パートナー　富士ソフト株式会社": {"industry": "IT", "revenue_size": "1000億円以上"},
    "導入パートナー　株式会社NTTデータ信越": {"industry": "IT", "revenue_size": "100億～500億円"},
    "導入パートナー　株式会社NTTデータ北海道": {"industry": "IT", "revenue_size": "100億～500億円"},
    "導入パートナー　株式会社サザンクロスシステムズ": {"industry": "IT", "revenue_size": "100億～500億円"},
    "導入パートナー　株式会社ビジネスブレイン太田昭和": {"industry": "IT", "revenue_size": "100億～500億円"},
    "導入パートナー　株式会社フォーカスシステムズ": {"industry": "IT", "revenue_size": "100億～500億円"},
    "導入パートナー　株式会社ＮＴＴデータ関西": {"industry": "IT", "revenue_size": "100億～500億円"},
    "導入パートナー　成都楷碼信息技術有限公司（成都楷碼）": {"industry": "IT", "revenue_size": "100億円未満"},
    # 海外企業
    "PT. ASURANSI TOKIO MARINE INDONESIA": {"industry": "保険", "revenue_size": "100億～500億円"},
    "北京蒲蒲蘭文化発展有限公司": {"industry": "小売", "revenue_size": "100億円未満"},
    "阿爾卑斯系統集成（大連）有限公司": {"industry": "IT", "revenue_size": "100億～500億円"},
}

def add_company_info_to_markdown(file_path, company_name):
    """Markdownファイルに業種と売上規模を追加"""
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
    if company_name in COMPANY_INFO:
        info = COMPANY_INFO[company_name]
    else:
        # 部分一致で検索
        for key, value in COMPANY_INFO.items():
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
    not_found_companies = set()
    
    for md_file in md_files:
        company_name = extract_company_name_from_markdown(md_file)
        if company_name:
            filename = os.path.basename(md_file)
            if add_company_info_to_markdown(md_file, company_name):
                print(f"✓ 更新: {filename}")
                updated_count += 1
            else:
                not_found_companies.add(company_name)
                not_found_count += 1
    
    print(f"\n更新したファイル数: {updated_count}")
    print(f"情報が見つからなかったファイル数: {not_found_count}")
    
    if not_found_companies:
        print(f"\n情報が見つからなかった会社（{len(not_found_companies)}社）:")
        for company in sorted(not_found_companies):
            print(f"  - {company}")

if __name__ == '__main__':
    main()
