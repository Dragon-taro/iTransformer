# iTransformer 推論サーバー

## 概要

このサーバーは、複数の時系列予測モデル（iTransformer）を同時に提供するAPIサービスです。設計目標は以下の通りです：

- 複数のモデル（通貨ペア・特徴量・アーキテクチャの違い）を単一のサーバーで提供
- モデル追加時の開発コストを最小化
- アプリケーションエンジニアとリサーチャの責務分離

## 起動方法

### 依存関係のインストール

```bash
pip install -r requirements.txt
```

### サーバーの起動

```bash
# 開発環境（自動リロード有効）
uvicorn src.server.main:app --reload --host 0.0.0.0 --port 8000

# 本番環境
uvicorn src.server.main:app --host 0.0.0.0 --port 8000 --workers 4
```

または、Pythonスクリプトから直接実行：

```bash
python -m src.server.main
```

## 主要なエンドポイント

### 共通エンドポイント

- `GET /` - サーバー状態とモデル一覧
- `GET /admin/models` - 利用可能なモデルの詳細情報
- `GET /admin/health` - サーバーと各モデルの健全性

### モデル固有エンドポイント

各モデルは `/{model_id}` のプレフィックスで以下のエンドポイントを提供：

- `GET /{model_id}` - モデル情報
- `POST /{model_id}/predict` - 単一予測の実行
- `POST /{model_id}/batch_predict` - バッチ予測の実行
- `WS /{model_id}/ws` - リアルタイム予測用WebSocket（開発中）

## 予測リクエストの例

```bash
curl -X POST "http://localhost:8000/usdjpy/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      [101.23, 101.45, 101.20, 101.30],
      [101.30, 101.50, 101.25, 101.40],
      ...
    ],
    "timestamps": [
      "2023-01-01T12:00:00Z",
      "2023-01-01T12:05:00Z",
      ...
    ]
  }'
```

## 新しいモデルの追加方法

1. `configs/models/{model_id}.yaml` に設定ファイルを作成
2. 必要に応じて `src/predictors/{model_id}.py` に特殊処理を実装（オプション）
3. サーバーを再起動すると自動的に新しいエンドポイントが追加される

## モデル設定ファイルの例

```yaml
# モデル識別情報
model_id: usdjpy
description: "USD/JPY為替レート予測モデル"
version: "1.0.0"

# モデル構成
model_type: iTransformer
checkpoint_path: "checkpoints/usdjpy/best_model.pth"
scaler_path: "artifacts/usdjpy/scaler.joblib"

# 入出力設定
features: ["Open", "High", "Low", "Close"]
target: ["Close"]
window_size: 60
pred_len: 10

# モデルパラメータ
d_model: 128
n_heads: 8
e_layers: 2
d_layers: 1
```

## 開発とデバッグ

サーバーのログは `logs/server.log` に出力されます。また、FastAPIの自動生成されたドキュメントには以下のURLでアクセスできます：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc 