# 環境変数の設定方法

## .envファイルを使用（推奨）

プロジェクトルートに`.env`ファイルを作成し、以下の内容を記述してください：

### 単一のAPIキーを使用する場合

```
OPENAI_API_KEY=sk-your-api-key-here
```

### 複数のAPIキーを使用する場合（並列リクエスト）

複数のAPIキーを指定すると、並列リクエストが可能になり、処理速度が向上し、レート制限を回避できます。以下の3つの方法から選択できます：

#### 方法1: 改行区切り（推奨）

`.env`ファイルで改行区切りで記述します：

```
OPENAI_API_KEYS=sk-key1-here
sk-key2-here
sk-key3-here
```

#### 方法2: カンマ区切り

1行でカンマ区切りで記述します：

```
OPENAI_API_KEYS=sk-key1-here,sk-key2-here,sk-key3-here
```

#### 方法3: 環境変数の複数定義

`OPENAI_API_KEY_1`, `OPENAI_API_KEY_2`などの形式で定義します：

```
OPENAI_API_KEY_1=sk-key1-here
OPENAI_API_KEY_2=sk-key2-here
OPENAI_API_KEY_3=sk-key3-here
```

**優先順位**: `OPENAI_API_KEYS`（改行区切り） > `OPENAI_API_KEYS`（カンマ区切り） > `OPENAI_API_KEY_N` > `OPENAI_API_KEY`

### 手順

1. `.env.example`をコピーして`.env`を作成：
   ```powershell
   Copy-Item .env.example .env
   ```

2. `.env`ファイルを開いて、`sk-your-api-key-here`を実際のAPIキーに置き換えてください

3. `.env`ファイルは`.gitignore`に含まれているため、Gitにコミットされません

### APIキーの取得方法

1. https://platform.openai.com/api-keys にアクセス
2. ログイン後、「Create new secret key」をクリック
3. 生成されたキーをコピー（`sk-`で始まる文字列）

## 確認

設定後、以下のコマンドで確認できます：

```powershell
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('API Key:', '設定済み' if os.getenv('OPENAI_API_KEY') else '未設定')"
```

## 注意事項

- `.env`ファイルは機密情報を含むため、Gitにコミットしないでください
- `.gitignore`に`.env`が含まれていることを確認してください
- チームで共有する場合は、`.env.example`を更新して、実際のキーは別途共有してください
