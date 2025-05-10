# iTransformer 推論サーバー

このサーバーはiTransformerモデルを使用して時系列予測を行うためのAPIを提供します。

## セットアップ

必要なパッケージをインストールしてください：

```bash
pip install flask torch numpy
```

## サーバーの起動

```bash
python app.py
```

サーバーはデフォルトで`http://0.0.0.0:5000`で実行されます。

## API エンドポイント

### ヘルスチェック

- エンドポイント: `/health`
- メソッド: `GET`
- 説明: サーバーの状態を確認します。
- レスポンス例:
  ```json
  {
    "status": "healthy"
  }
  ```

### 予測実行

- エンドポイント: `/predict`
- メソッド: `POST`
- 説明: モデルを使用して予測を実行します。
- リクエストボディ: モデルと予測のパラメータを含むJSON
- レスポンス: 予測結果とメトリクスを含むJSON

## リクエスト例

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "test",
    "model": "iTransformer",
    "data": "ETTh1",
    "features": "M",
    "seq_len": 96,
    "pred_len": 48,
    "enc_in": 7,
    "dec_in": 7,
    "c_out": 7,
    "d_model": 512,
    "n_heads": 8,
    "e_layers": 2,
    "d_layers": 1,
    "checkpoints": "./checkpoints/"
  }'
```

## パラメータ説明

リクエストJSONには以下のパラメータを含めることができます：

### 必須パラメータ
- `model_id`: モデルID
- `model`: モデル名（iTransformer, iInformer, iReformer, iFlowformer, iFlashformer）
- `data`: データセットタイプ

### データロード関連
- `root_path`: データファイルのルートパス
- `data_path`: データCSVファイル
- `features`: 予測タスク（M, S, MS）
- `target`: ターゲット特徴（SまたはMSタスクの場合）
- `freq`: 時間特徴エンコーディングの頻度
- `checkpoints`: モデルチェックポイントの場所

### 予測タスク
- `seq_len`: 入力シーケンス長
- `label_len`: 開始トークン長
- `pred_len`: 予測シーケンス長

### モデル定義
- `enc_in`: エンコーダ入力サイズ
- `dec_in`: デコーダ入力サイズ
- `c_out`: 出力サイズ
- `d_model`: モデルの次元
- `n_heads`: ヘッド数
- `e_layers`: エンコーダ層数
- `d_layers`: デコーダ層数
- `d_ff`: FCN次元
- `class_strategy`: クラス戦略（projection/average/cls_token）

他のパラメータについては、`app.py`の`get_args`関数を参照してください。 