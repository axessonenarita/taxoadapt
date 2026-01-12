# 環境変数の設定方法

## .envファイルを使用（推奨）

プロジェクトルートに`.env`ファイルを作成し、以下の内容を記述してください：

```
OPENAI_API_KEY=sk-your-api-key-here
```

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
