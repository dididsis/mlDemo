# ベースイメージ
FROM python:3.9-slim

# 作業ディレクトリ
WORKDIR /app

# 必要なファイルをコピー
COPY requirements.txt requirements.txt
COPY app.py app.py

# パッケージのインストール
RUN pip install --no-cache-dir -r requirements.txt

# ポートを公開
EXPOSE 8501

# Streamlitアプリの起動
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]