#!/usr/bin/env python3
"""
TaxoAdapt ツール CLI
"""
import argparse
import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.company_info_manager import generate_company_info_json, add_company_info_to_markdown
from tools.markdown_fixer import fix_all_markdown_files
from tools.cost_estimator import estimate_cost, print_estimate
from tools.markdown_utils import get_all_markdown_files


def main():
    parser = argparse.ArgumentParser(description='TaxoAdapt ツール集')
    subparsers = parser.add_subparsers(dest='command', help='実行するコマンド')
    
    # 会社情報JSON生成
    parser_generate = subparsers.add_parser('generate-json', help='company_info.jsonを生成')
    parser_generate.add_argument('--output', '-o', default='datasets/casestudy/company_info.json',
                                help='出力ファイルパス')
    
    # Markdownファイル修正
    parser_fix = subparsers.add_parser('fix-markdown', help='Markdownファイルの構造を修正')
    parser_fix.add_argument('--directory', '-d', default='assets/casestudy',
                           help='Markdownファイルのディレクトリ')
    parser_fix.add_argument('--no-partner', action='store_true',
                           help='導入パートナーの修正をスキップ')
    parser_fix.add_argument('--no-structure', action='store_true',
                           help='構造の修正をスキップ')
    
    # コスト概算
    parser_cost = subparsers.add_parser('estimate-cost', help='実行コストを概算')
    parser_cost.add_argument('--papers', '-p', type=int, default=96,
                            help='論文数')
    parser_cost.add_argument('--dimensions', '-d', type=int, default=7,
                            help='ディメンション数')
    parser_cost.add_argument('--max-depth', type=int, default=2,
                            help='最大深度')
    parser_cost.add_argument('--init-levels', type=int, default=1,
                            help='初期レベル')
    parser_cost.add_argument('--max-density', type=int, default=40,
                            help='最大密度')
    
    # リスト表示
    parser_list = subparsers.add_parser('list', help='Markdownファイルの一覧を表示')
    parser_list.add_argument('--directory', '-d', default='assets/casestudy',
                            help='Markdownファイルのディレクトリ')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'generate-json':
        generate_company_info_json(args.output)
    
    elif args.command == 'fix-markdown':
        fix_all_markdown_files(
            directory=args.directory,
            fix_partner=not args.no_partner,
            fix_structure=not args.no_structure
        )
    
    elif args.command == 'estimate-cost':
        result = estimate_cost(
            num_papers=args.papers,
            num_dimensions=args.dimensions,
            max_depth=args.max_depth,
            init_levels=args.init_levels,
            max_density=args.max_density
        )
        print_estimate(result)
    
    elif args.command == 'list':
        md_files = get_all_markdown_files(args.directory)
        print(f"Markdownファイル数: {len(md_files)}")
        for md_file in md_files:
            print(f"  - {os.path.basename(md_file)}")


if __name__ == '__main__':
    main()
