import os
import numpy as np
import pandas as pd
import requests
import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import joblib  # joblibライブラリをインポート
from sklearn.preprocessing import MinMaxScaler  # MinMaxScalerをインポート

class USDJPYPredictionAPI:
    def __init__(self, server_url='http://localhost:5001', base_data_dir=None):
        """USDJPYの予測APIクライアント

        Args:
            server_url (str): 推論サーバーのURL
            base_data_dir (str, optional): 前処理済みデータディレクトリのパス
        """
        self.server_url = server_url
        
        if base_data_dir is None:
            # デフォルトのデータディレクトリパス
            self.base_data_dir = os.path.join(
                os.path.dirname(__file__), 
                'scripts/multivariate_forecasting/USDJPY/pretreatment/pretreatment_data'
            )
        else:
            self.base_data_dir = base_data_dir
            
        # スケーラーを初期化
        self.scaler = self._load_scaler()
        
    def _load_scaler(self):
        """スケーラーを読み込む"""
        # スケーラーファイルパス
        scaler_joblib_path = os.path.join(self.base_data_dir, 'usdjpy_scaler.joblib')
        npz_path = os.path.join(self.base_data_dir, 'usdjpy_windows.npz')
        
        # joblibで保存されたスケーラーを探す
        if os.path.exists(scaler_joblib_path):
            try:
                scaler = joblib.load(scaler_joblib_path)
                print(f"joblibスケーラーを読み込みました: {scaler_joblib_path}")
                return scaler
            except Exception as e:
                print(f"joblibスケーラーの読み込みに失敗しました: {e}")
        
        # npzファイルからスケーラーを再構築
        if os.path.exists(npz_path):
            try:
                npz_data = np.load(npz_path)
                scaler = MinMaxScaler()
                scaler.data_min_ = npz_data['scaler_min']
                scaler.data_max_ = npz_data['scaler_max']
                scaler.scale_ = 1.0 / (scaler.data_max_ - scaler.data_min_)
                print(f"npzファイルからスケーラーを再構築しました: {npz_path}")
                return scaler
            except Exception as e:
                print(f"npzファイルからのスケーラー再構築に失敗しました: {e}")
        
        print("スケーラーが見つかりませんでした。元のスケールでの表示ができません。")
        return None
    
    def load_test_data(self, n_samples=1, file_name='usdjpy_X_test_wdate.csv', offset=0):
        """テストデータを読み込み、iTransformer用の形式に変換

        Args:
            n_samples (int): 読み込むサンプル数
            file_name (str): 読み込むファイル名
            offset (int): データの開始位置のオフセット

        Returns:
            tuple: (dates, data) - 日時とiTransformer用の入力データ
        """
        file_path = os.path.join(self.base_data_dir, file_name)
        print(f"データ読み込み中: {file_path}")
        
        # ファイルが存在するか確認
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"データファイルが見つかりません: {file_path}")
        
        # 予測期間のデータも含めて読み込む
        y_test_file = file_name.replace('X_test', 'y_test')
        y_test_path = os.path.join(self.base_data_dir, y_test_file)
        
        # オフセットを適用してデータを読み込み
        # ヘッダー行は保持したまま、データ行のみをスキップ
        df = pd.read_csv(file_path, skiprows=range(1, offset + 1), nrows=n_samples)
        
        # 日付列を抽出
        dates = pd.to_datetime(df['date'])
        
        # 特徴量の数を取得
        # 最初の列はdate, 残りの列はOpen,High,Low,Closeがwindow_size回繰り返される
        columns = df.columns[1:]  # date列を除外
        
        # カラム名からfeature名を抽出して、ユニークな特徴量数を計算
        feature_names = []
        for col in columns:
            # 'Open', 'High', 'Low', 'Close'などの特徴量名を抽出
            if col.split('.')[0] in ['Open', 'High', 'Low', 'Close']:
                feature_names.append(col.split('.')[0])
            else:
                feature_names.append(col)
        
        unique_features = list(set(feature_names))
        n_features = len(unique_features)
        window_size = len(columns) // n_features
        
        print(f"特徴量数: {n_features}, ウィンドウサイズ: {window_size}")
        
        # データを整形
        data = df.iloc[:, 1:].values  # date列を除外
        
        # iTransformer用の形式に変換 [samples, seq_len, features]
        data_reshaped = []
        for i in range(len(data)):
            sample = []
            for j in range(window_size):
                features = []
                for k in range(n_features):
                    idx = k * window_size + j
                    if idx < data.shape[1]:
                        features.append(data[i, idx])
                    else:
                        # インデックスが範囲外の場合、0で埋める
                        features.append(0.0)
                sample.append(features)
            data_reshaped.append(sample)
        
        data_array = np.array(data_reshaped)
        print(f"変換後のデータ形状: {data_array.shape}")
        
        # 予測期間のデータを読み込む
        if os.path.exists(y_test_path):
            print(f"予測期間のデータを読み込み中: {y_test_path}")
            y_df = pd.read_csv(y_test_path, skiprows=range(1, offset + 1), nrows=n_samples)
            if 'date' in y_df.columns:
                y_data = y_df.drop(columns=['date']).values
                print(f"予測期間のデータ形状: {y_data.shape}")
                return dates, data_array, y_data
            
        return dates, data_array, None
    
    def predict(self, data, pred_len=10):
        """推論サーバーに予測リクエストを送信

        Args:
            data (numpy.ndarray): [samples, seq_len, features]形式のデータ
            pred_len (int): 予測長さ

        Returns:
            dict: レスポンス（予測結果とメトリクス）
        """
        # リクエストデータの作成
        seq_len, n_features = data.shape[1], data.shape[2]
        
        # 最小限のリクエストデータ
        request_data = {
            "input_data": data.tolist(),
            "seq_len": seq_len,
            "pred_len": pred_len,
            "enc_in": n_features
        }
        
        # リクエスト送信
        print(f"推論サーバーにリクエスト送信中: {self.server_url}/predict")
        try:
            # タイムアウトを設定して接続エラーを早期に検出
            response = requests.post(f"{self.server_url}/predict", json=request_data, timeout=30)
            
            # ステータスコードのチェック
            print(f"ステータスコード: {response.status_code}")
            
            # エラーレスポンスの場合はその内容を表示
            if response.status_code != 200:
                print(f"エラーレスポンス: {response.text}")
                
            # raise_for_statusでエラーを発生させる
            response.raise_for_status()
            
            # レスポンスのJSONパース
            try:
                result = response.json()
                print(f"レスポンスの概要: {list(result.keys()) if isinstance(result, dict) else '辞書型ではありません'}")
                return result
            except json.JSONDecodeError as e:
                print(f"JSONデコードエラー: {e}")
                print(f"レスポンスの内容: {response.text[:200]}...")  # 先頭200文字のみ表示
                return None
                
        except requests.exceptions.ConnectionError as e:
            print(f"接続エラー: {e}")
            print("サーバーが起動しているか、URLが正しいか確認してください")
            return None
        except requests.exceptions.Timeout as e:
            print(f"タイムアウトエラー: {e}")
            print("サーバーの処理に時間がかかりすぎているか、応答がありません")
            return None
        except requests.exceptions.RequestException as e:
            print(f"リクエストエラー: {e}")
            print(f"レスポンスの内容: {e.response.text if hasattr(e, 'response') and e.response else '利用不可'}")
            return None
    
    def predict_and_suggest_trade(self, data, pred_len=10):
        """推論サーバーに予測と取引戦略提案のリクエストを送信

        Args:
            data (numpy.ndarray): [samples, seq_len, features]形式のデータ
            pred_len (int): 予測長さ

        Returns:
            dict: レスポンス（予測結果とトレード提案）
        """
        # リクエストデータの作成
        seq_len, n_features = data.shape[1], data.shape[2]
        
        # 最小限のリクエストデータ
        request_data = {
            "input_data": data.tolist(),
            "seq_len": seq_len,
            "pred_len": pred_len,
            "enc_in": n_features
        }
        
        # リクエスト送信
        print(f"推論サーバーにトレード提案リクエスト送信中: {self.server_url}/predict_and_suggest_trade")
        try:
            # タイムアウトを設定して接続エラーを早期に検出
            response = requests.post(f"{self.server_url}/predict_and_suggest_trade", json=request_data, timeout=30)
            
            # ステータスコードのチェック
            print(f"ステータスコード: {response.status_code}")
            
            # エラーレスポンスの場合はその内容を表示
            if response.status_code != 200:
                print(f"エラーレスポンス: {response.text}")
                
            # raise_for_statusでエラーを発生させる
            response.raise_for_status()
            
            # レスポンスのJSONパース
            try:
                result = response.json()
                print(f"レスポンスの概要: {list(result.keys()) if isinstance(result, dict) else '辞書型ではありません'}")
                return result
            except json.JSONDecodeError as e:
                print(f"JSONデコードエラー: {e}")
                print(f"レスポンスの内容: {response.text[:200]}...")  # 先頭200文字のみ表示
                return None
                
        except requests.exceptions.ConnectionError as e:
            print(f"接続エラー: {e}")
            print("サーバーが起動しているか、URLが正しいか確認してください")
            return None
        except requests.exceptions.Timeout as e:
            print(f"タイムアウトエラー: {e}")
            print("サーバーの処理に時間がかかりすぎているか、応答がありません")
            return None
        except requests.exceptions.RequestException as e:
            print(f"リクエストエラー: {e}")
            print(f"レスポンスの内容: {e.response.text if hasattr(e, 'response') and e.response else '利用不可'}")
            return None
    
    def plot_prediction(self, dates, original_data, predictions, y_data=None, feature_idx=3, output_file='usdjpy_prediction.png'):
        """予測結果をプロット
        
        Args:
            dates (pandas.Series): 日時
            original_data (numpy.ndarray): 元データ
            predictions (numpy.ndarray): 予測結果（すでにスケーラー適用済み）
            y_data (numpy.ndarray, optional): 予測期間の実データ
            feature_idx (int): 使用する特徴量のインデックス（3=Close価格）
            output_file (str): 出力する画像ファイル名
        """
        # 予測データを適切な形式に変換
        if isinstance(predictions, list):
            predictions = np.array(predictions)
        
        # オリジナルデータの形状を確認
        print(f"オリジナルデータの形状: {original_data.shape}")
        batch_size, seq_len, n_features = original_data.shape
        
        # 予測データの形状を確認
        if len(predictions.shape) == 3 and predictions.shape[2] > 1:
            # 予測結果の中からClose価格（feature_idx）を抽出
            pred_values = predictions[0, :, feature_idx]
        elif len(predictions.shape) == 3:
            # 予測結果が [batch, pred_len, 1] の場合
            pred_values = predictions[0, :, 0]
        else:
            # 予測結果が [batch, pred_len] の場合
            pred_values = predictions[0]
        
        pred_len = len(pred_values)  # 予測長を取得
        
        # 元のスケールに戻す処理（ヒストリカルデータのみ）
        original_close_scaled = original_data[0, :, feature_idx].flatten()
        original_close = None
        
        if self.scaler is not None:
            try:
                print("スケーラーを使用してヒストリカルデータを逆変換します")
                
                # データの最小値と最大値を確認
                data_min = self.scaler.data_min_
                data_max = self.scaler.data_max_
                print(f"スケーラーの最小値: {data_min}")
                print(f"スケーラーの最大値: {data_max}")
                
                # ヒストリカルデータのみを逆変換
                dummy_hist = np.zeros((len(original_close_scaled), n_features))
                for i in range(len(original_close_scaled)):
                    for j in range(n_features):
                        if j == feature_idx:
                            dummy_hist[i, j] = original_close_scaled[i]
                        else:
                            # 他の特徴量は元データからコピー
                            dummy_hist[i, j] = original_data[0, i, j]
                
                # 逆変換
                hist_inverted = self.scaler.inverse_transform(dummy_hist)
                original_close = hist_inverted[:, feature_idx]
                
                # 予測期間の実データがある場合は逆変換
                if y_data is not None:
                    # y_dataの形状を確認
                    y_data = np.array(y_data).reshape(-1)  # 1次元配列に変換
                    dummy_y = np.zeros((len(y_data), n_features))
                    dummy_y[:, feature_idx] = y_data
                    y_data_inverted = self.scaler.inverse_transform(dummy_y)[:, feature_idx]
                    print(f"予測期間の実データ（最初/最後）: {y_data_inverted[0]:.4f}/{y_data_inverted[-1]:.4f}")
                
                print(f"元のスケールに戻したヒストリカルデータ（最初/最後）: {original_close[0]:.4f}/{original_close[-1]:.4f}")
                print(f"予測データ（最初/最後）: {pred_values[0]:.4f}/{pred_values[-1]:.4f}")
            except Exception as e:
                print(f"スケール変換中にエラーが発生しました: {e}")
                import traceback
                traceback.print_exc()
                original_close = original_close_scaled
                if y_data is not None:
                    y_data_inverted = y_data
        else:
            # スケーラーがない場合はそのまま使用
            print("スケーラーが利用できないため、ヒストリカルデータはスケール変換なしで使用します")
            original_close = original_close_scaled
            if y_data is not None:
                y_data_inverted = y_data
        
        # プロット
        plt.figure(figsize=(12, 6))
        
        # 時間軸の設定
        x_original = np.arange(len(original_close))  # ヒストリカルデータ用の時間軸
        x_pred = np.arange(60, 60 + pred_len)  # 予測用の時間軸（60分からスタート）
        
        # データの値域を確認
        print(f"ヒストリカルデータの範囲: {np.min(original_close):.4f} ~ {np.max(original_close):.4f}")
        print(f"予測データの範囲: {np.min(pred_values):.4f} ~ {np.max(pred_values):.4f}")
        
        # オリジナルデータをプロット
        plt.plot(x_original, original_close, 'b-', label='Historical Data')
        
        # 予測値をプロット
        plt.plot(x_pred, pred_values, 'r-', label='Prediction')
        
        # 過去データの最後の点から予測の最初の点までをグレーの線で接続
        plt.plot([x_original[-1], x_pred[0]], [original_close[-1], pred_values[0]], 
                 'gray', linestyle='--', alpha=0.5)
        
        # 予測期間の実データがある場合はプロット
        if y_data is not None:
            plt.plot(x_pred, y_data_inverted, 'g--', label='Actual Future')
            # 過去データの最後の点から実データの最初の点までをグレーの線で接続
            plt.plot([x_original[-1], x_pred[0]], [original_close[-1], y_data_inverted[0]], 
                     'gray', linestyle='--', alpha=0.5)
        
        # 軸ラベルとタイトル
        plt.xlabel('Time Steps (minutes)')
        plt.ylabel('Price')
        plt.title(f'USDJPY Prediction from {dates[0]}')
        plt.legend()
        plt.grid(True)
        
        # 保存
        plt.savefig(output_file)
        plt.close()
        print(f"予測結果プロットを保存しました: {output_file}")

    def find_volatile_periods(self, file_name='usdjpy_X_test_wdate.csv', price_range_threshold=0.5, n_th_volatile=1):
        """価格変動幅が大きい期間のオフセットを見つける

        Args:
            file_name (str): 読み込むファイル名
            price_range_threshold (float): 価格変動幅の閾値（元のスケールでの値）
            n_th_volatile (int): N番目の価格変動幅の大きい期間を選択

        Returns:
            int: 選択された期間の開始オフセット
        """
        file_path = os.path.join(self.base_data_dir, file_name)
        
        # データ全体を読み込む
        df = pd.read_csv(file_path)
        
        # Close価格の列を特定（60分のウィンドウ分）
        close_columns = [col for col in df.columns if col.startswith('Close')]
        if len(close_columns) < 60:
            raise ValueError("60分のClose価格データが見つかりません")
        
        # 各行について60分間の価格変動幅を計算
        price_ranges = []
        for index, row in df.iterrows():
            # 60分のClose価格を取得
            close_prices = [row[col] for col in close_columns[:60]]
            
            # スケーラーが利用可能な場合、元のスケールに戻す
            if self.scaler is not None:
                dummy_data = np.zeros((60, 4))  # 4 features (OHLC)
                dummy_data[:, 3] = close_prices  # Close価格
                close_prices = self.scaler.inverse_transform(dummy_data)[:, 3]
            
            # 価格変動幅を計算（最大値 - 最小値）
            price_range = np.max(close_prices) - np.min(close_prices)
            price_ranges.append(price_range)
        
        # 価格変動幅が閾値を超える期間を特定
        volatile_periods = []
        for i, price_range in enumerate(price_ranges):
            if price_range > price_range_threshold:
                volatile_periods.append(i)
        
        if not volatile_periods:
            print(f"閾値 {price_range_threshold} を超える価格変動幅の期間が見つかりませんでした")
            return 0
        
        # N番目の価格変動幅の大きい期間を選択
        if n_th_volatile > len(volatile_periods):
            print(f"指定された {n_th_volatile} 番目の期間が存在しません。最後の期間を使用します。")
            selected_offset = volatile_periods[-1]
        else:
            selected_offset = volatile_periods[n_th_volatile - 1]
        
        print(f"選択された期間のオフセット: {selected_offset}")
        print(f"この期間の価格変動幅: {price_ranges[selected_offset]:.6f}")
        
        return selected_offset


def main():
    """コマンドラインからの実行用エントリーポイント"""
    parser = argparse.ArgumentParser(description='USDJPY予測API')
    parser.add_argument('--server', type=str, default='http://localhost:5001', help='推論サーバーのURL')
    parser.add_argument('--samples', type=int, default=1, help='使用するサンプル数')
    parser.add_argument('--sample_skip', type=int, default=1, help='サンプル間のスキップ数（デフォルト: 1）')
    parser.add_argument('--pred_len', type=int, default=10, help='予測する時間ステップ数')
    parser.add_argument('--data_dir', type=str, default=None, help='データディレクトリのパス')
    parser.add_argument('--file', type=str, default='usdjpy_X_test_wdate.csv', help='使用するデータファイル')
    parser.add_argument('--feature_idx', type=int, default=3, help='使用する特徴量のインデックス（0=Open, 1=High, 2=Low, 3=Close）')
    parser.add_argument('--offset', type=int, default=0, help='データの開始位置のオフセット')
    parser.add_argument('--price_range', type=float, default=None, 
                       help='価格変動幅の閾値（元のスケールでの値）。指定した場合、この閾値を超える期間を選択します。')
    parser.add_argument('--nth_range', type=int, default=1,
                       help='N番目の価格変動幅の大きい期間を選択（デフォルト: 1）')
    parser.add_argument('--suggest_trade', action='store_true', help='取引戦略の提案を取得する')
    args = parser.parse_args()
    
    try:
        # APIクライアントを初期化
        api = USDJPYPredictionAPI(server_url=args.server, base_data_dir=args.data_dir)
        
        # サンプル数分ループ処理
        for sample_idx in range(args.samples):
            print(f"\n処理サンプル {sample_idx + 1}/{args.samples}")
            
            # 価格変動幅ベースでオフセットを選択
            base_offset = args.offset + (sample_idx * args.sample_skip)  # サンプルごとにスキップ数を考慮したオフセット
            if args.price_range is not None:
                offset = api.find_volatile_periods(
                    file_name=args.file,
                    price_range_threshold=args.price_range,
                    n_th_volatile=args.nth_range + (sample_idx * args.sample_skip)  # サンプルごとにスキップ数を考慮
                )
            else:
                offset = base_offset
            
            print(f"使用するオフセット: {offset}")
            
            # テストデータを読み込み
            dates, data, y_data = api.load_test_data(n_samples=1, file_name=args.file, offset=offset)
            print(f"データ形状: {data.shape}")
            
            # 予測実行
            print("予測リクエスト送信中...")
            if args.suggest_trade:
                response = api.predict_and_suggest_trade(data, pred_len=args.pred_len)
                if response:
                    # 予測結果とトレード提案を表示
                    predictions = np.array(response['prediction']['predictions'])
                    trade_suggestion = response['trade_suggestion']
                    print(f"予測形状: {predictions.shape}")
                    print("\n取引提案:")
                    print(f"アクション: {trade_suggestion['action']}")
                    print(f"信頼度: {trade_suggestion['confidence']:.2f}")
                    print(f"エントリー価格: {trade_suggestion['entry_price']:.3f}")
                    if trade_suggestion['target_price']:
                        print(f"目標価格: {trade_suggestion['target_price']:.3f}")
                    if trade_suggestion['stop_loss']:
                        print(f"ストップロス: {trade_suggestion['stop_loss']:.3f}")
                    if 'risk_management' in trade_suggestion:
                        print("\nリスク管理提案:")
                        for key, value in trade_suggestion['risk_management'].items():
                            print(f"- {key}: {value}")
            else:
                response = api.predict(data, pred_len=args.pred_len)
            
            if response:
                # 予測結果を表示
                predictions = np.array(response['predictions'])
                print(f"予測形状: {predictions.shape}")
                
                # タイムスタンプとサンプルインデックスを含むファイル名を生成
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_filename = f"usdjpy_prediction_{timestamp}_sample{sample_idx + 1}"
                
                # 結果をプロット
                plot_filename = f"{base_filename}.png"
                api.plot_prediction(dates, data, predictions, y_data=y_data, 
                                 feature_idx=args.feature_idx, output_file=plot_filename)
                
                # 予測結果を保存
                result_file = f"{base_filename}.json"
                with open(result_file, 'w') as f:
                    result_data = {
                        'date': str(dates[0]),
                        'prediction': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
                        'metrics': response.get('metrics', {}),
                        'sample_index': sample_idx + 1,
                        'offset': offset,
                        'sample_skip': args.sample_skip
                    }
                    json.dump(result_data, f, indent=2)
                print(f"予測結果を保存しました: {result_file}")
            else:
                print(f"サンプル {sample_idx + 1} の推論サーバーからの応答がありませんでした")
    
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 