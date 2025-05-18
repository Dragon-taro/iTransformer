# マルチモデル対応 iTransformer サービス設計ドキュメント

## 1. 目標

- **複数モデル** (通貨ペア・特徴量・アーキテクチャ違い) を共通基盤の上で運用できるようにする。
- **前処理・学習・推論サーバー** の重複コードを排除し、**追加コスト最小**で新モデルを導入可能とする。
- アプリケーションエンジニアとリサーチャが **責務分離** されたワークフローで協働できるようにする。

## 2. 現状の課題

| 層 | 課題内容 |
|----|----------|
| 前処理 | 通貨ペアごと・パラメータごとにスクリプトがファット化し、共通処理と個別処理が混在して読みづらい。 |
| 推論サーバー | app.py が 1,000 行超。モデル固有ロジックと HTTP エンドポイントがハードコーディングされており、モデル追加時に大量のコピペが発生する。 |
| prediction_api.py | サーバーとフォーマット依存コードが肥大化し、用途が混在。 |

## 3. 全体アーキテクチャ

```text
┌────────────────────────────┐
│   CLI / Notebook / Frontend │
└────────┬───────────────────┘
         │❶ REST / gRPC
┌────────▼───────────────────┐           Config (YAML / json)
│        Inference Server     │◄──────────────────────────────┐
│  (FastAPI + Uvicorn)        │           ❹ Model Registry    │
│                              │                               │
│  ┌──────────────┐            │  ❷ load()                    │
│  │  Router/      │  Factory  │ ─────────────► BasePredictor │
│  │  Controller   │──────────►│                (abstract)   │
│  └───▲───────────┘           │              ▲               │
│      │                       │              │implements     │
│  ┌───┴───────────┐           │              │               │
│  │ DataProvider  │◄──────────┘   USDJPYPredictor, ETHJPY…   │
│  └───────────────┘
└──────────────────────────────┘
             ▲ ❸ train() / eval()
┌────────────┴───────────────┐
│       Training Pipeline     │ (PyTorch-Lightning)
└─────────────────────────────┘
```

### フロー概要
1. **モデル登録 (Model Registry) :** YAML でモデル ID, チェックポイント, 前処理設定, ハイパーパラメータ, 使用エンコーダ/デコーダを記述。
2. **Inference Server** 起動時にレジストリをロードし、エンドポイントを自動生成。
3. 推論リクエスト時に `model_id` で **Predictor Factory** が該当クラスを動的インポートし、共有 `BasePredictor` インターフェースで実行。
4. Predictor 内部でスケーラー・データプロバイダ・iTransformer などを構築。
5. 新しいモデルは **YAML 1 ファイル & Predictor サブクラス (≈50 行)** を追加するだけで API 提供可能。

## 4. ディレクトリ構成案

```text
project/
├─ configs/
│   ├─ models/
│   │   ├─ usdjpy.yaml
│   │   └─ btcusd.yaml
│   └─ data/
│       └─ usdjpy_pretreat.yaml
├─ scripts/
│   └─ preprocessing/
│       ├─ base.py          # 共通ロジック
│       └─ usdjpy.py        # 個別設定 (必要なら)
├─ src/
│   ├─ predictors/
│   │   ├─ base.py          # BasePredictor (抽象クラス)
│   │   └─ usdjpy.py        # サブクラス
│   ├─ server/
│   │   ├─ main.py          # FastAPI エントリ
│   │   └─ router_factory.py
│   └─ training/
│       ├─ datamodule.py    # PyTorch-Lightning
│       └─ train.py
└─ tests/
```

## 5. 前処理レイヤ (scripts/preprocessing)

- `BasePreprocessor` に **ファイルロード・欠損補完・リサンプリング** を実装。
- 派生クラスで **列名差分や特殊処理** のみ override。
- 出力は `(npz, joblib, csv)` を共通フォーマットに統一。
- YAML で window_size, target_size,  frequency などを管理し、再現性を確保。

## 6. 学習パイプライン

- **PyTorch-Lightning** で `LightningModule` + `LightningDataModule` を採用し、学習・検証・テストを一本化。
- `configs/models/*.yaml` に hyper-params を記述し CLI (`python train.py --config configs/models/usdjpy.yaml`) で実行。
- Checkpoint は `artifacts/{model_id}/{timestamp}/` に保存してレジストリに自動追記。

## 7. 推論サーバー詳細

### 7.1 BasePredictor 抽象メソッド

| メソッド | 役割 |
|----------|------|
| `load_model()` | チェックポイント読み込み (strict=False 対応) |
| `prepare_input(raw_json)` | JSON → Tensor 変換、スケール変換 |
| `predict(batch)` | forward 実行し numpy 返却 |
| `postprocess(pred)` | inverse_scale, 取引シグナル生成など |

### 7.2 Router 自動生成

```python
from fastapi import APIRouter
from predictors.factory import get_predictor

def create_router(model_id: str):
    router = APIRouter()

    @router.post("/predict")
    async def predict(payload: dict):
        predictor = get_predictor(model_id)
        return predictor.run(payload)

    return router
```

`main.py` で YAML を走査して以下を登録:

```python
for cfg_path in Path("configs/models").glob("*.yaml"):
    model_id = cfg_path.stem
    app.include_router(create_router(model_id), prefix=f"/{model_id}")
```

### 7.3 追加手順
1. `configs/models/xxx.yaml` を置く。
2. 必要なら `predictors/xxx.py` を `BasePredictor` 継承で作成。
3. サーバー再起動 → `/xxx/predict` エンドポイントが自動公開。

## 8. prediction_api リファクタリング

- クライアントはシンプルな **Facade** とし、`model_id` を指定するだけで使用。
- プロット・ボラティリティ分析などのユーティリティは `notebooks/utils` へ切り出す。

```python
api = ForecastClient(server="http://localhost:8000")
resp = api.predict(model_id="usdjpy", data=array)
```

## 9. テスト戦略

- `pytest` + `fastapi.testclient` で **Predictor 単体** と **API 統合** を CI 実行。
- サンプル入力を fixture にし、出力 shape とスケール逆変換の整合性を検証。

## 10. 今後の拡張余地

- **gRPC / WebSocket** でのストリーミング推論対応。
- **Model Versioning** (MLflow, Weights & Biases) との連携。
- **オンライン学習** / バックテスト基盤の統合。

---

以上の設計により、モデル追加は **YAML + 小規模クラス** のみで済み、既存コードの肥大化を防ぎながらメンテナンス性と拡張性を向上させることができます。
