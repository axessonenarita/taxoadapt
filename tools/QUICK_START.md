# クイックスタートガイド

## 基本的な使い方

### 1. company_info.jsonを生成

```bash
python -m tools.cli generate-json
```

すべてのMarkdownファイルから会社情報を抽出し、`datasets/casestudy/company_info.json`を生成します。

### 2. Markdownファイルの構造を修正

```bash
python -m tools.cli fix-markdown
```

以下の問題を自動的に修正します：
- `company_name`がない
- `company_name`の値が不正（長すぎる、説明文になっている）
- 最初のタイトルがない
- 導入パートナーが`company_name`になっている
- フロントマターの改行が不正

### 3. コストを概算

```bash
python -m tools.cli estimate-cost
```

TaxoAdapt実行時のコストを概算します。

### 4. Markdownファイルの一覧を表示

```bash
python -m tools.cli list
```

処理対象のMarkdownファイルの一覧を表示します。

## オプション

### generate-json

```bash
python -m tools.cli generate-json --output datasets/casestudy/company_info.json
```

### fix-markdown

```bash
# 導入パートナーの修正をスキップ
python -m tools.cli fix-markdown --no-partner

# 構造の修正をスキップ
python -m tools.cli fix-markdown --no-structure

# 別のディレクトリを指定
python -m tools.cli fix-markdown --directory assets/blog
```

### estimate-cost

```bash
# カスタムパラメータで概算
python -m tools.cli estimate-cost --papers 100 --dimensions 5 --max-depth 3
```

## Pythonから直接使用

```python
from tools.company_info_manager import generate_company_info_json
from tools.markdown_fixer import fix_all_markdown_files
from tools.cost_estimator import estimate_cost, print_estimate

# 会社情報JSONを生成
generate_company_info_json()

# Markdownファイルを修正
fix_all_markdown_files()

# コストを概算
result = estimate_cost(num_papers=96, num_dimensions=7)
print_estimate(result)
```
