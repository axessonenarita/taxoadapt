# アーカイブされたスクリプト

このディレクトリには、`tools/`ディレクトリに統合された古いスクリプトが保存されています。

## 統合先

以下のスクリプトは`tools/`ディレクトリに統合されました：

- `extract_company_names.py` → `tools/markdown_utils.py`
- `generate_company_info_json.py` → `tools/company_info_manager.py`
- `add_company_info_batch.py` → `tools/company_info_manager.py`
- `add_company_info.py` → `tools/company_info_manager.py` (古いバージョン)
- `check_and_fix_markdown_structure.py` → `tools/markdown_fixer.py`
- `fix_partner_company_names.py` → `tools/markdown_fixer.py`
- `fix_frontmatter_newlines.py` → `tools/markdown_fixer.py`
- `fix_markdown_titles.py` → `tools/markdown_fixer.py` (一部機能)
- `convert_markdown_to_frontmatter.py` → 使用済み（一度だけ実行、参考用として保持）
- `estimate_cost.py` → `tools/cost_estimator.py`

## 使用方法

新しいツールは`tools/cli.py`から使用できます：

```bash
# company_info.jsonを生成
python -m tools.cli generate-json

# Markdownファイルの構造を修正
python -m tools.cli fix-markdown

# コストを概算
python -m tools.cli estimate-cost
```

詳細は`tools/README.md`を参照してください。
