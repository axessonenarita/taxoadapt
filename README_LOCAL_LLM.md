# ローカルLLM（LM Studio/Ollama）セットアップガイド

このガイドでは、RTX 5090（32GB VRAM）でQwen3-VL-30Bモデルを使用するためのセットアップ手順を説明します。

## 概要

LM StudioまたはOllamaのOpenAI互換APIを使用することで、ローカル環境で大規模言語モデルを実行できます。Qwen3-VL-30BはAlibabaが開発したVision Languageモデルで、RTX 5090（32GB VRAM）で動作可能です。LM Studioではモデル名として `qwen/qwen3-vl-30b` が使用されます。

## 前提条件

- **GPU**: RTX 5090（32GB VRAM）または同等のGPU
- **OS**: Windows 10/11
- **Python**: 3.8以上

## セットアップ手順

### オプション1: LM Studioを使用する場合

1. **LM Studioのインストール**
   - [LM Studio公式サイト](https://lmstudio.ai/)からダウンロードしてインストール

2. **モデルファイルのダウンロード**
   - LM Studioを起動
   - 検索バーで「qwen3-vl-30b」または「Qwen3-VL-30B」を検索
   - Hugging Faceからモデルを選択（GGUF形式またはHugging Face形式）
   - ダウンロード（モデルサイズは形式により異なります）
   - モデル名は `qwen/qwen3-vl-30b` として表示される場合があります

3. **モデルのロードと設定**
   - ダウンロードしたモデルを選択
   - **Context Length**: `40960`（最大対応）または必要に応じて調整
   - **GPU Offload**: `Max`（全レイヤー）に設定
   - 「Load」をクリックしてモデルをロード

4. **OpenAI互換APIサーバーの起動**
   - 左側のメニューから「Local Server」を選択
   - 「Start Server」をクリック
   - デフォルトで `http://localhost:1234/v1` で起動

### オプション2: Ollamaを使用する場合

1. **Ollamaのインストール**
   - [Ollama公式サイト](https://ollama.ai/)からダウンロードしてインストール

2. **モデルファイルの準備**
   - OllamaでQwen3 30B A3Bを使用する場合、Hugging Face形式のモデルを使用
   - または、Ollamaがサポートする形式のモデルを使用
   - モデル名: `Qwen3-30B-A3B` または `qwen3:30b-a3b`

3. **OpenAI互換APIの有効化**
   - Ollamaはデフォルトで `http://localhost:11434/v1` でOpenAI互換APIを提供

## 環境変数の設定

`.env`ファイルに以下を設定：

```env
# LM Studioを使用する場合
LOCAL_API_URL=http://localhost:1234/v1
LOCAL_API_KEY=lm-studio
LOCAL_MODEL_NAME=qwen/qwen3-vl-30b

# Ollamaを使用する場合
# LOCAL_API_URL=http://localhost:11434/v1
# LOCAL_API_KEY=ollama
# LOCAL_MODEL_NAME=qwen/qwen3-vl-30b
```

## 実行方法

### コマンドライン引数を使用する場合

```bash
# LM Studioを使用
python main.py --llm local \
  --local_api_url http://localhost:1234/v1 \
  --local_model_name qwen/qwen3-vl-30b

# Ollamaを使用
python main.py --llm local \
  --local_api_url http://localhost:11434/v1 \
  --local_model_name qwen/qwen3-vl-30b
```

### 環境変数を使用する場合

`.env`ファイルに設定済みの場合：

```bash
python main.py --llm local
```

## パフォーマンス

Qwen3 30B A3Bモデルの特徴：

- **モデルタイプ**: MoE（Mixture-of-Experts）モデル
- **GPUメモリ**: 最低16GB以上推奨（RTX 5090 32GBで動作可能）
- **コンテキスト長**: 最大40,960トークンに対応
- **思考モード**: 複雑な推論が必要な場合は思考モードをオンに可能
- **多言語対応**: 100以上の言語に対応

## トラブルシューティング

### エラー: 接続できない

- LM Studio/Ollamaが起動しているか確認
- ポート番号が正しいか確認（LM Studio: 1234, Ollama: 11434）
- ファイアウォールの設定を確認

### エラー: モデルが見つからない

- モデル名が正しいか確認
- LM Studio/Ollamaでモデルがロードされているか確認
- モデル名は大文字小文字を区別する場合があります

### VRAM不足エラー

- コンテキスト長を減らす（必要に応じて）
- 他のアプリケーションを終了してVRAMを確保
- モデルの量子化レベルを確認（必要に応じて）

## 参考情報

- **モデル**: Qwen3-VL-30B（Alibaba開発のVision Languageモデル）
- **モデル名（LM Studio）**: `qwen/qwen3-vl-30b`
- **モデルファイル**: Hugging FaceまたはModelScopeからダウンロード可能
- **コンテキスト長**: 最大40,960トークン
- **特徴**: 思考モード切り替え、多言語対応、長文処理、画像理解

## 注意事項

- LM Studio/Ollamaが起動している必要があります
- モデルファイルがロードされている必要があります
- コンテキスト長はLM Studio/Ollama側で設定してください（最大40,960トークン）
- 初回実行時はモデルのロードに時間がかかります
- Qwen3 30B A3BはMoEモデルのため、推論時のGPUメモリ使用量に注意してください
