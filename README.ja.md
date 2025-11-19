# 🤖 LLMベースの求人マッチングツール  
*（Anyone AI Machine Learning Engineering プログラムの一環）*

大規模言語モデル（LLM）を活用し、求職者のレジュメ内容に基づいて最適な求人をレコメンドするインテリジェントな就職支援アプリケーションです。

## 📋 概要

本ツールは **LangChain**, **ベクトル埋め込み**, **RAG（検索拡張生成）**, **ChromaDB** を組み合わせた AI 求人マッチングシステムです。  
PDFレジュメから候補者情報を抽出し、セマンティック検索により適切な求人を提示します。

### 主な機能

- **マルチアシスタント構成**
  - **Vanilla ChatGPT**：汎用対話アシスタント  
  - **Jobs Finder Assistant**：レジュメに基づく求人セマンティック検索  
  - **Jobs Agent**：ツール利用可能な上位エージェント、カバーレター自動生成も対応  

- **レジュメ解析**（PDFテキスト抽出 & 要約）
- **セマンティック検索**（ChromaDBによる埋め込み検索）
- **複数LLMプロバイダ対応**（OpenAI / Google Gemini）
- **対話履歴管理**（メモリ搭載）
- **ChainlitによるインタラクティブUI**

## 🏗️ アーキテクチャ

```
├── backend/
│   ├── app.py                          # メインのChainlitアプリ
│   ├── config.py                       # 設定管理
│   ├── etl.py                          # ベクトルDB構築ETL
│   ├── llm_factory.py                  # LLMプロバイダ切替用ファクトリ
│   ├── retriever.py                    # 求人検索ロジック
│   ├── utils.py                        # PDF処理などのユーティリティ
│   └── models/
│       ├── chatgpt_clone.py           # 一般会話モデル
│       ├── jobs_finder.py             # 求人マッチング用モデル
│       ├── jobs_finder_agent.py       # 上位エージェントモデル
│       └── resume_summarizer_chain.py # レジュメ要約チェーン
├── dataset/
│   └── jobs.csv                        # 求人データ
├── tests/                              # テスト
├── .chainlit/                          # Chainlit設定
├── docker-compose.yml                  # Docker構成
├── Dockerfile                          # Dockerfile
└── requirements.txt                    # 依存パッケージ
```

## 🛠️ 技術スタック

- **LLMフレームワーク**：LangChain  
- **ベクトルDB**：ChromaDB  
- **埋め込みモデル**：sentence-transformers (paraphrase-MiniLM-L6-v2)  
- **LLM**：OpenAI GPT-4o-mini / Google Gemini 2.5 Flash  
- **UI**：Chainlit  
- **PDF解析**：PyPDF2  
- **データ処理**：Pandas / NumPy  
- **コンテナ**：Docker / Docker Compose  

## 🚀 クイックスタート

### 前提条件

- Docker & Docker Compose（推奨）  
**または**
- Python 3.11+
- Git

---

## 🐳 Docker を使う（推奨）

### Step 1: ビルド & 起動
```bash
docker-compose up --build
```

### Step 2: 初回のみ、ベクトルDBを初期化
```bash
docker-compose exec llm-recruitment-tool python backend/etl.py
```

### Step 3: ブラウザでアクセス  
```
http://localhost:8000
```

### 停止
```bash
docker-compose down
```

---

## 💻 ローカル実行（非Docker）

### Step 1: 仮想環境
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### Step 2: 依存インストール
```bash
pip install -r requirements.txt
```

### Step 3: ベクトルDB作成
```bash
python backend/etl.py
```

### Step 4: Server起動
```bash
chainlit run backend/app.py
```

---

## 📊 動作フロー

1. **データ取り込み**：`etl.py` が jobs.csv を処理 → 埋め込み作成 → ChromaDBに格納  
2. **レジュメアップロード**  
3. **PDF解析 & 要約**  
4. **セマンティック検索で求人マッチング**  
5. **LLMによる結果生成**（求人推薦 / カバーレター生成など）

---

## 🧪 テスト

```bash
pytest tests/
```

特定のテスト:
```bash
pytest tests/backend/test_utils.py
pytest tests/backend/models/test_chatgpt_clone.py
```

Dockerでテスト:
```bash
docker-compose exec llm-recruitment-tool python -m pytest tests/
```

---

## 🎨 コードスタイル

Blackでフォーマット：
```bash
black --line-length=88 .
```

---

## 🔧 LLMプロバイダ切替

`.env` で以下を変更するだけでLLMを切替可能：

```
LLM_PROVIDER="gemini"  # または "openai"
```

---

## 🐛 トラブルシューティング

### Port already in use
→ `docker-compose.yml` のポート変更 or 使用中プロセスを停止。

### API Key Error
- `.env` のAPI Keyを確認  
- `LLM_PROVIDER` が一致しているかチェック

### ベクトルDBが空
→ `python backend/etl.py` を再実行  
→ `dataset/jobs.csv` の存在を確認

---

## 📄 ライセンス

Anyone AI Machine Learning Engineering プログラム教材の一部です。

---

## 🤝 コントリビューション

Pull Request 歓迎！

---

## 📚 参考リンク

- LangChain Docs  
- Chainlit Docs  
- ChromaDB Docs  
- Docker Docs  
