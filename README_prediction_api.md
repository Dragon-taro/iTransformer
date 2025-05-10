# USDJPY予測API

このAPIは、前処理済みのUSDJPYデータを使用して、iTransformer推論サーバーに予測リクエストを送信するためのものです。

## セットアップ

必要なパッケージをインストールしてください：

```bash
pip install pandas numpy requests matplotlib scikit-learn
```

## 使用方法

### 1. 推論サーバーの起動

まず、iTransformer推論サーバーを起動します：

```bash
python app.py
```

サーバーはデフォルトで`http://localhost:5000`で実行されます。

### 2. 予測APIの実行

次に、予測APIを実行します：

```bash
python prediction_api.py --samples 1 --pred_len 10
```

#### オプション

- `--server`: 推論サーバーのURL（デフォルト: http://localhost:5000）
- `--samples`: 使用するサンプル数（デフォルト: 1）
- `--pred_len`: 予測する時間ステップ数（デフォルト: 10）
- `--data_dir`: データディレクトリのパス（デフォルト: scripts/multivariate_forecasting/USDJPY/pretreatment/pretreatment_data）
- `--file`: 使用するデータファイル（デフォルト: usdjpy_X_test_wdate.csv）

例：

```bash
python prediction_api.py --server http://localhost:5000 --samples 1 --pred_len 30 --file usdjpy_X_test_wdate.csv
```

### 3. 出力

実行すると、以下の出力が生成されます：

1. 予測結果のプロット画像: `usdjpy_prediction.png`
2. 予測結果のJSONファイル: `usdjpy_prediction_[timestamp].json`

## APIの仕組み

1. 前処理済みのUSDJPYデータからテストサンプルを読み込みます
2. データをiTransformer用の形式に変換します
3. 推論サーバーに予測リクエストを送信します
4. 結果を受け取り、元のスケールに戻します（可能な場合）
5. 予測結果をプロットし、JSONファイルに保存します

## プログラムの構成

- `USDJPYPredictionAPI`: メインのAPIクラス
  - `__init__`: APIクライアントの初期化
  - `load_test_data`: テストデータの読み込み
  - `predict`: 推論サーバーへの予測リクエスト送信
  - `inverse_transform`: 予測結果を元のスケールに戻す
  - `plot_prediction`: 予測結果のプロット
- `main`: コマンドラインからの実行用エントリーポイント

## 予測サーバーの拡張機能

`app.py`は以下の拡張機能を持っています：

1. JSONリクエストから直接入力データを受け取る機能
2. カスタムデータプロバイダを使用した柔軟なデータ処理
3. 予測結果の返却とモデル管理

## トラブルシューティング

エラーが発生した場合は、以下を確認してください：

1. 推論サーバーが起動しているか
2. データパスが正しいか
3. モデルチェックポイントが存在するか

詳細なエラーメッセージを確認し、必要に応じて問題を修正してください。 