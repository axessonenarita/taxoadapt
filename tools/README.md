# TaxoAdapt ツール集

Markdownファイルの処理、会社情報の管理、コスト概算などのユーティリティツール集です。

## 構成

- `markdown_utils.py`: Markdownファイル処理の基本ユーティリティ
- `company_info_manager.py`: 会社情報の抽出・管理
- `markdown_fixer.py`: Markdownファイルの構造修正
- `cost_estimator.py`: 実行コストの概算
- `cli.py`: コマンドラインインターフェース

## 使用方法

### CLI経由（推奨）

```bash
# company_info.jsonを生成
python -m tools.cli generate-json

# Markdownファイルの構造を修正
python -m tools.cli fix-markdown

# コストを概算
python -m tools.cli estimate-cost

# Markdownファイルの一覧を表示
python -m tools.cli list
```

### 直接インポート

```python
from tools.company_info_manager import generate_company_info_json
from tools.markdown_fixer import fix_all_markdown_files
from tools.cost_estimator import estimate_cost, print_estimate

# 使用例
generate_company_info_json('datasets/casestudy/company_info.json')
fix_all_markdown_files()
result = estimate_cost(num_papers=96, num_dimensions=7)
print_estimate(result)
```

## 各ツールの説明

### markdown_utils.py

Markdownファイル処理の基本機能を提供します。

- `extract_company_name_from_markdown()`: ファイルから会社名を抽出
- `extract_company_name_from_content()`: コンテンツから会社名を抽出
- `extract_metadata_from_markdown()`: YAMLフロントマターからメタデータを抽出
- `fix_frontmatter_newlines()`: フロントマターの改行を修正
- `get_all_markdown_files()`: すべてのMarkdownファイルを取得

### company_info_manager.py

会社情報の抽出・管理を行います。

- `generate_company_info_json()`: すべてのMarkdownファイルから会社情報を抽出してJSONを生成
- `add_company_info_to_markdown()`: Markdownファイルに会社情報を追加

### markdown_fixer.py

Markdownファイルの構造をチェック・修正します。

- `check_and_fix_markdown_structure()`: ファイルの構造をチェックし、問題があれば修正
- `fix_partner_company_names()`: 導入パートナーがcompany_nameになっているファイルを修正
- `fix_all_markdown_files()`: すべてのMarkdownファイルを一括修正

### cost_estimator.py

TaxoAdapt実行時のコストを概算します。

- `estimate_cost()`: コストを概算
- `print_estimate()`: 概算結果を表示

## 移行されたスクリプト

以下のスクリプトは`tools/`ディレクトリに統合されました：

- `extract_company_names.py` → `tools/markdown_utils.py`
- `generate_company_info_json.py` → `tools/company_info_manager.py`
- `add_company_info_batch.py` → `tools/company_info_manager.py`
- `check_and_fix_markdown_structure.py` → `tools/markdown_fixer.py`
- `fix_partner_company_names.py` → `tools/markdown_fixer.py`
- `fix_frontmatter_newlines.py` → `tools/markdown_fixer.py`
- `estimate_cost.py` → `tools/cost_estimator.py`

古いスクリプトは`scripts_archive/`に移動されました。参考用として保持しています。

## 新しい機能を追加する際の注意

新しいPythonファイルを追加する前に、`DEVELOPMENT_GUIDELINES.md`を参照してください。
既存のツールに統合できる機能は、新しいファイルを作成せずに既存ツールに追加することを推奨します。
