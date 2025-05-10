from flask import Flask, request, jsonify
import torch
import argparse
import random
import numpy as np
import time  # 追加
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from experiments.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial
import os
from sklearn.preprocessing import MinMaxScaler
import joblib  # joblibをインポート

app = Flask(__name__)

class CustomDataProvider:
    """カスタムデータプロバイダクラス"""
    def __init__(self, input_data, seq_len, pred_len, enc_in, scale=True, external_scaler=None):
        """
        Args:
            input_data (numpy.ndarray): 入力データ、形状 [batch_size, seq_len, features]
            seq_len (int): シーケンス長
            pred_len (int): 予測長
            enc_in (int): 特徴量数
            scale (bool): スケーリングするかどうか
            external_scaler: 外部から渡されるスケーラーオブジェクト
        """
        self.input_data = input_data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.scale = scale
        
        # スケーラー情報
        self.scaler = external_scaler  # 外部から渡されたスケーラーを最初に設定
        self.scaler_min = None
        self.scaler_max = None
        
        # 外部スケーラーが設定されている場合はそれを優先使用
        if self.scaler is not None:
            print("外部から渡されたスケーラーを使用します")
        else:
            # スケーラーを探す
            base_path = os.path.join(os.path.dirname(__file__), 'scripts/multivariate_forecasting/USDJPY/pretreatment/pretreatment_data')
            scaler_joblib_path = os.path.join(base_path, 'usdjpy_scaler.joblib')
            npz_path = os.path.join(base_path, 'usdjpy_windows.npz')
            
            print(f"スケーラー検索パス: {base_path}")
            print(f"joblib スケーラーパス存在: {os.path.exists(scaler_joblib_path)}")
            print(f"npz ファイルパス存在: {os.path.exists(npz_path)}")
            
            if scale:
                # 1. まず、joblibで保存されたスケーラーを探す
                if os.path.exists(scaler_joblib_path):
                    try:
                        self.scaler = joblib.load(scaler_joblib_path)
                        print("joblibスケーラーファイルを読み込みました")
                    except Exception as e:
                        print(f"joblibスケーラーファイルの読み込みに失敗しました: {e}")
                        self.scaler = None
                
                # 2. joblibスケーラーが見つからない場合は、npzから再構築を試みる
                if self.scaler is None and os.path.exists(npz_path):
                    try:
                        npz_data = np.load(npz_path)
                        self.scaler_min = npz_data['scaler_min']
                        self.scaler_max = npz_data['scaler_max']
                        self.scaler = MinMaxScaler()
                        self.scaler.data_min_ = self.scaler_min
                        self.scaler.data_max_ = self.scaler_max
                        self.scaler.scale_ = 1.0 / (self.scaler_max - self.scaler_min)
                        print("npzファイルからスケーラー情報を再構築しました")
                    except Exception as e:
                        print(f"npzファイルからのスケーラー再構築に失敗しました: {e}")
                        self.scaler = None
                
                # 3. いずれの方法でもスケーラーが見つからない場合
                if self.scaler is None:
                    print(f"スケーラーファイルが見つかりません: {scaler_joblib_path} または {npz_path}")
                    print("スケーリングなしで実行します")
            else:
                print("スケーリングが無効化されています")
        
        # ダミー出力データ（予測用なので実際は使わない）
        self.batch_size = input_data.shape[0]
        self.dummy_output = np.zeros((self.batch_size, self.pred_len, self.enc_in))
        
        # バッチインデックス
        self.batch_idx = 0
        
    def __len__(self):
        """データセットの長さを返す"""
        return 1  # 1バッチのみ
        
    def __iter__(self):
        """イテレータを返す"""
        self.batch_idx = 0
        return self
        
    def __next__(self):
        """次のバッチを返す"""
        if self.batch_idx < 1:
            # 入力データ用のタイムインデックス（時間特徴量）
            # [batch, seq_len] または [batch, seq_len, feat_dim] 形式が必要
            batch_size = self.input_data.shape[0]
            seq_len = self.input_data.shape[1]
            pred_len = self.pred_len
            
            # 時間情報を特徴量として扱う（3次元テンソルとして生成）
            timestamp_feat_dim = 4  # 時間特徴の次元数（例：時、分、曜日、月）
            
            # 時間特徴（ここでは単なるインデックス）を生成
            # 入力用の時間特徴 [batch, seq_len, feat_dim]
            batch_x_mark = np.zeros((batch_size, seq_len, timestamp_feat_dim))
            for i in range(batch_size):
                for j in range(seq_len):
                    # 単純な位置エンコーディング（実際にはもっと複雑なものが必要かもしれない）
                    batch_x_mark[i, j, 0] = j / seq_len  # 正規化された位置
                    batch_x_mark[i, j, 1] = np.sin(j * (2 * np.pi / seq_len))  # sin位置
                    batch_x_mark[i, j, 2] = np.cos(j * (2 * np.pi / seq_len))  # cos位置
                    batch_x_mark[i, j, 3] = (j % 7) / 7.0  # 週ごとのパターン
            
            # 出力用の時間特徴 [batch, pred_len, feat_dim]
            batch_y_mark = np.zeros((batch_size, pred_len, timestamp_feat_dim))
            for i in range(batch_size):
                for j in range(pred_len):
                    # 予測部分の位置エンコーディング
                    pos = seq_len + j
                    batch_y_mark[i, j, 0] = pos / (seq_len + pred_len)  # 正規化された位置
                    batch_y_mark[i, j, 1] = np.sin(pos * (2 * np.pi / (seq_len + pred_len)))
                    batch_y_mark[i, j, 2] = np.cos(pos * (2 * np.pi / (seq_len + pred_len)))
                    batch_y_mark[i, j, 3] = (pos % 7) / 7.0  # 週ごとのパターン
            
            # テンソルに変換
            batch_x = torch.FloatTensor(self.input_data)
            batch_y = torch.FloatTensor(self.dummy_output)
            batch_x_mark = torch.FloatTensor(batch_x_mark)
            batch_y_mark = torch.FloatTensor(batch_y_mark)
            
            # 形状を確認（デバッグ用）
            print(f"データ形状: batch_x={batch_x.shape}, batch_y={batch_y.shape}")
            print(f"マーク形状: batch_x_mark={batch_x_mark.shape}, batch_y_mark={batch_y_mark.shape}")
            
            self.batch_idx += 1
            return batch_x, batch_y, batch_x_mark, batch_y_mark
        else:
            raise StopIteration
    
    def inverse_transform(self, predictions, feature_idx=3):
        """予測結果を元のスケールに戻す
        
        Args:
            predictions (numpy.ndarray): 予測結果
            feature_idx (int): 使用する特徴量のインデックス（3=Close価格）
            
        Returns:
            numpy.ndarray: 元のスケールに戻した予測結果
        """
        if self.scaler is None:
            print("スケーラーが設定されていないため、逆変換は行わずそのまま返します")
            return predictions
        
        try:
            # 予測結果の次元数に応じて処理を分岐
            if len(predictions.shape) == 3:
                # [batch, time, features] の場合
                # Close価格のみを抽出
                predictions_close = predictions[:, :, feature_idx] if predictions.shape[2] > feature_idx else predictions[:, :, 0]
                
                # ダミーデータを作成して逆変換
                dummy = np.zeros((predictions_close.shape[0] * predictions_close.shape[1], len(self.scaler.data_min_)))
                dummy_idx = 0
                
                for i in range(predictions_close.shape[0]):
                    for j in range(predictions_close.shape[1]):
                        dummy[dummy_idx, feature_idx] = predictions_close[i, j]
                        dummy_idx += 1
                
                # 逆変換 - スケーラーのinverse_transformメソッドを直接使用
                dummy_inverted = self.scaler.inverse_transform(dummy)
                
                # 元の形状に戻す
                result = np.zeros_like(predictions_close)
                dummy_idx = 0
                for i in range(predictions_close.shape[0]):
                    for j in range(predictions_close.shape[1]):
                        result[i, j] = dummy_inverted[dummy_idx, feature_idx]
                        dummy_idx += 1
                
                return result
            else:
                # 単一サンプルの場合
                dummy = np.zeros((len(predictions), len(self.scaler.data_min_)))
                dummy[:, feature_idx] = predictions
                dummy_inverted = self.scaler.inverse_transform(dummy)
                return dummy_inverted[:, feature_idx]
        except Exception as e:
            print(f"逆変換中にエラーが発生しました: {e}")
            print("元のデータをそのまま返します")
            return predictions

def get_args(request_json):
    """リクエストJSONからargsを作成"""
    args = argparse.Namespace()
    
    # 必須パラメータ
    args.is_training = 0  # 推論モード
    args.model_id = request_json.get('model_id', 'usdjpy')
    args.model = request_json.get('model', 'iTransformer')
    args.data = request_json.get('data', 'custom')
    
    # データロード関連
    args.root_path = request_json.get('root_path', './data/electricity/')
    args.data_path = request_json.get('data_path', 'electricity.csv')
    args.features = request_json.get('features', 'M')
    args.target = request_json.get('target', 'Close')
    args.freq = request_json.get('freq', 'h')
    args.checkpoints = request_json.get('checkpoints', './checkpoints/')
    
    # 予測タスク
    args.seq_len = request_json.get('seq_len', 96)
    args.label_len = request_json.get('label_len', 48)
    args.pred_len = request_json.get('pred_len', 96)
    
    # モデル定義
    args.enc_in = request_json.get('enc_in', 7)
    args.dec_in = request_json.get('dec_in', 7)
    args.c_out = request_json.get('c_out', 1)
    args.d_model = request_json.get('d_model', 128)
    args.n_heads = request_json.get('n_heads', 8)
    args.e_layers = request_json.get('e_layers', 2)
    args.d_layers = request_json.get('d_layers', 1)
    args.d_ff = request_json.get('d_ff', 128)
    args.moving_avg = request_json.get('moving_avg', 25)
    args.factor = request_json.get('factor', 1)
    args.distil = request_json.get('distil', True)
    args.dropout = request_json.get('dropout', 0.1)
    args.embed = request_json.get('embed', 'timeF')
    args.activation = request_json.get('activation', 'gelu')
    args.output_attention = request_json.get('output_attention', False)
    args.do_predict = request_json.get('do_predict', False)
    
    # 最適化
    args.num_workers = request_json.get('num_workers', 10)
    args.itr = request_json.get('itr', 1)
    args.train_epochs = request_json.get('train_epochs', 10)
    args.batch_size = request_json.get('batch_size', 32)
    args.patience = request_json.get('patience', 3)
    args.learning_rate = request_json.get('learning_rate', 0.0001)
    args.des = request_json.get('des', 'test')
    args.loss = request_json.get('loss', 'MSE')
    args.lradj = request_json.get('lradj', 'type1')
    args.use_amp = request_json.get('use_amp', False)
    
    # GPU
    args.use_gpu = torch.cuda.is_available() and request_json.get('use_gpu', True)
    args.gpu = request_json.get('gpu', 0)
    args.use_multi_gpu = request_json.get('use_multi_gpu', False)
    args.devices = request_json.get('devices', '0,1,2,3')
    
    # iTransformer
    args.exp_name = request_json.get('exp_name', 'MTSF')
    args.channel_independence = request_json.get('channel_independence', False)
    args.inverse = request_json.get('inverse', True)  # デフォルトで逆変換を有効化
    args.class_strategy = request_json.get('class_strategy', 'projection')
    args.target_root_path = request_json.get('target_root_path', './data/electricity/')
    args.target_data_path = request_json.get('target_data_path', 'electricity.csv')
    args.efficient_training = request_json.get('efficient_training', False)
    args.use_norm = request_json.get('use_norm', True)
    args.partial_start_index = request_json.get('partial_start_index', 0)
    
    # スケーリング設定
    args.scale = request_json.get('scale', True)
    args.feature_idx = request_json.get('feature_idx', 3)  # 3=Close
    
    # GPUの設定
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    return args

@app.route('/predict', methods=['POST'])
def predict():
    """予測API"""
    # 開始時間を記録
    start_time = time.time()
    
    # デバッグフラグ - 詳細なログを出力するかどうか
    DEBUG = True
    
    # リクエストJSONを取得
    request_json = request.get_json()
    if not request_json:
        return jsonify({"error": "リクエストボディが必要です"}), 400
    
    # シード固定
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    
    try:
        # 引数を取得
        args = get_args(request_json)
        
        # 実験クラスを設定
        if args.exp_name == 'partial_train':
            Exp = Exp_Long_Term_Forecast_Partial
        else:
            Exp = Exp_Long_Term_Forecast
        
        # 設定文字列を作成
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            args.class_strategy, ii)
        
        # モデルチェックポイントのパスを探す
        checkpoint_path = f'./checkpoints/{setting}/checkpoint.pth'
        
        # 既定のパスでチェックポイントが見つからない場合、既存のチェックポイントを探す
        if not os.path.exists(checkpoint_path):
            print(f"指定された設定でチェックポイントが見つかりません: {checkpoint_path}")
            
            # ./checkpoints ディレクトリ内のサブディレクトリを探す
            if os.path.exists('./checkpoints'):
                available_checkpoints = [d for d in os.listdir('./checkpoints') 
                                        if os.path.isdir(os.path.join('./checkpoints', d)) and 
                                        os.path.exists(os.path.join('./checkpoints', d, 'checkpoint.pth'))]
                
                # model_idを含むディレクトリを優先して探す
                model_id_checkpoints = [d for d in available_checkpoints if args.model_id in d]
                
                if model_id_checkpoints:
                    # model_idを含むチェックポイントがある場合は最初のものを使用
                    checkpoint_dir = model_id_checkpoints[0]
                    print(f"model_id '{args.model_id}' を含むチェックポイントが見つかりました: {checkpoint_dir}")
                    setting = checkpoint_dir
                    checkpoint_path = f'./checkpoints/{setting}/checkpoint.pth'
                elif available_checkpoints:
                    # model_idを含むものがなければ最初のチェックポイントを使用
                    checkpoint_dir = available_checkpoints[0]
                    print(f"利用可能なチェックポイントが見つかりました: {checkpoint_dir}")
                    setting = checkpoint_dir
                    checkpoint_path = f'./checkpoints/{setting}/checkpoint.pth'
                else:
                    return jsonify({
                        "error": "利用可能なモデルチェックポイントが見つかりません。",
                        "setting": setting
                    }), 404
            else:
                return jsonify({
                    "error": "checkpointsディレクトリが見つかりません。",
                    "setting": setting
                }), 404
        
        print(f"使用するチェックポイント: {checkpoint_path}")
        
        # チェックポイントの設定からモデルパラメータを抽出
        # 例: usdjpy_60_10_iTransformer_custom_M_ft60_sl48_ll10_pl128_dm8_nh2_el1_dl128_df1_fctimeF_ebTrue_dtExp_projection_0
        try:
            checkpoint_dir = os.path.basename(os.path.dirname(checkpoint_path))
            print(f"チェックポイントディレクトリ: {checkpoint_dir}")
            
            # モデルパラメータの抽出
            parts = checkpoint_dir.split('_')
            for i, part in enumerate(parts):
                if part == 'dm' and i+1 < len(parts):
                    args.d_model = int(parts[i+1])
                    print(f"チェックポイントから抽出したd_model: {args.d_model}")
                elif part == 'df' and i+1 < len(parts):
                    args.d_ff = int(parts[i+1])
                    print(f"チェックポイントから抽出したd_ff: {args.d_ff}")
                elif part == 'el' and i+1 < len(parts):
                    args.e_layers = int(parts[i+1])
                    print(f"チェックポイントから抽出したe_layers: {args.e_layers}")
                elif part == 'nh' and i+1 < len(parts):
                    args.n_heads = int(parts[i+1])
                    print(f"チェックポイントから抽出したn_heads: {args.n_heads}")
        except Exception as e:
            print(f"チェックポイント名からのパラメータ抽出中にエラー: {e}")
            print("抽出に失敗しても処理を続行します")
        
        # 実験を設定
        exp = Exp(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        
        # 入力データがリクエストに含まれている場合
        input_data = request_json.get('input_data')
        if input_data is not None:
            # 入力データをnumpyに変換
            input_data = np.array(input_data)
            print(f"Input data shape: {input_data.shape}")
            
            # 入力データがスケーリングされていないか確認
            is_scaled = request_json.get('is_scaled', False)
            
            # データがスケーリングされていない場合、スケーリングを行う
            if not is_scaled:
                print("入力データがスケーリングされていないため、スケーリングを行います")
                
                # スケーラーを取得
                scaler = None
                base_path = os.path.join(os.path.dirname(__file__), 'scripts/multivariate_forecasting/USDJPY/pretreatment/pretreatment_data')
                scaler_joblib_path = os.path.join(base_path, 'usdjpy_scaler.joblib')
                npz_path = os.path.join(base_path, 'usdjpy_windows.npz')
                
                # 1. まず、joblibで保存されたスケーラーを探す
                if os.path.exists(scaler_joblib_path):
                    try:
                        scaler = joblib.load(scaler_joblib_path)
                        print("joblibスケーラーファイルを読み込みました")
                    except Exception as e:
                        print(f"joblibスケーラーファイルの読み込みに失敗しました: {e}")
                        scaler = None
                
                # 2. joblibスケーラーが見つからない場合は、npzから再構築を試みる
                if scaler is None and os.path.exists(npz_path):
                    try:
                        npz_data = np.load(npz_path)
                        scaler_min = npz_data['scaler_min']
                        scaler_max = npz_data['scaler_max']
                        scaler = MinMaxScaler()
                        scaler.data_min_ = scaler_min
                        scaler.data_max_ = scaler_max
                        scaler.scale_ = 1.0 / (scaler_max - scaler_min)
                        print("npzファイルからスケーラー情報を再構築しました")
                    except Exception as e:
                        print(f"npzファイルからのスケーラー再構築に失敗しました: {e}")
                        scaler = None
                
                # スケーラーが見つかった場合、入力データをスケーリング
                if scaler is not None:
                    # 入力データの形状を保存
                    original_shape = input_data.shape
                    
                    # データを2Dに変形 (samples, features)
                    reshaped_data = input_data.reshape(-1, original_shape[-1])
                    
                    # スケーリング実行
                    scaled_data = scaler.transform(reshaped_data)
                    
                    # 元の形状に戻す
                    input_data = scaled_data.reshape(original_shape)
                    print("入力データをスケーリングしました")
                else:
                    print("スケーラーが見つからないため、スケーリングなしで実行します")
            
            # カスタムデータローダーを作成 (スケーリング済みデータを使用)
            data_provider = CustomDataProvider(
                input_data=input_data,
                seq_len=args.seq_len,
                pred_len=args.pred_len,
                enc_in=args.enc_in,
                scale=False  # すでにスケーリング済みか、スケーラーが見つからなかった場合
            )
            
            # 重要: スケーラーを明示的に設定（予測値を元のスケールに戻すため）
            if 'scaler' in locals() and scaler is not None:
                data_provider.scaler = scaler
                print("データプロバイダにスケーラーを明示的に設定しました")
            
            # 推論実行
            predictions = []
            predictions_original = []
            try:
                print(f"モデルロード中: {checkpoint_path}")
                # strictをFalseに設定して、モデル構造が完全に一致しなくても読み込めるようにする
                model_state = torch.load(checkpoint_path, map_location=torch.device('cpu' if not args.use_gpu else 'cuda'))
                
                # モデル構造の詳細をログに出力（デバッグ用）
                print(f"モデルの状態キー: {list(model_state.keys())[:5]} ...")
                
                # 不一致を許容してロード
                exp.model.load_state_dict(model_state, strict=False)
                print("モデルのロードに成功しました（strict=False）")
                exp.model.eval()
                
                with torch.no_grad():
                    for batch_x, batch_y, batch_x_mark, batch_y_mark in data_provider:
                        batch_x = batch_x.float().to(exp.device)
                        batch_y = batch_y.float().to(exp.device)
                        
                        if 'PEMS' in args.data or 'Solar' in args.data:
                            batch_x_mark = None
                            batch_y_mark = None
                        else:
                            batch_x_mark = batch_x_mark.float().to(exp.device)
                            batch_y_mark = batch_y_mark.float().to(exp.device)
                        
                        # decoder input
                        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(exp.device)
                        
                        print(f"モデル入力形状: batch_x={batch_x.shape}, batch_x_mark={batch_x_mark.shape if batch_x_mark is not None else None}")
                        print(f"デコーダ入力形状: dec_inp={dec_inp.shape}, batch_y_mark={batch_y_mark.shape if batch_y_mark is not None else None}")
                        
                        # encoder - decoder
                        if args.use_amp:
                            with torch.cuda.amp.autocast():
                                if args.output_attention:
                                    outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            if args.output_attention:
                                outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        
                        f_dim = -1 if args.features == 'MS' else 0
                        outputs = outputs[:, -args.pred_len:, f_dim:]
                        outputs = outputs.detach().cpu().numpy()
                        
                        # 元のスケールに戻す（ユーザー指定に従う）
                        if args.inverse and hasattr(data_provider, 'inverse_transform'):
                            outputs_original = data_provider.inverse_transform(outputs, feature_idx=args.feature_idx)
                            predictions_original.append(outputs_original)
                            print("予測結果を元のスケールに戻しました")
                        else:
                            predictions.append(outputs)
                            print("予測結果はスケーリングされたままです")
            except Exception as model_error:
                import traceback
                error_trace = traceback.format_exc()
                return jsonify({
                    "error": f"モデル推論中にエラーが発生しました: {str(model_error)}",
                    "traceback": error_trace
                }), 500
            
            # 予測結果を結合
            predictions = np.concatenate(predictions, axis=0)
            predictions_original = np.concatenate(predictions_original, axis=0)
            
            # スケール前後の値の範囲をログに出力（デバッグ用）
            if len(predictions.shape) == 3 and predictions.shape[2] > args.feature_idx:
                # マルチ特徴量の場合
                print(f"スケール後の予測値範囲(Close価格): {np.min(predictions[0, :, args.feature_idx]):.6f}～{np.max(predictions[0, :, args.feature_idx]):.6f}")
                print(f"元のスケールに戻した予測値範囲: {np.min(predictions_original):.6f}～{np.max(predictions_original):.6f}")
            else:
                # 単一特徴量の場合
                print(f"スケール後の予測値範囲: {np.min(predictions):.6f}～{np.max(predictions):.6f}")
                print(f"元のスケールに戻した予測値範囲: {np.min(predictions_original):.6f}～{np.max(predictions_original):.6f}")
            
            # GPU キャッシュをクリア
            torch.cuda.empty_cache()
            
            # 終了時間を記録し、経過時間を計算（ミリ秒単位）
            end_time = time.time()
            inference_time_ms = (end_time - start_time) * 1000
            print(f"推論時間: {inference_time_ms:.2f}ms")
            
            # 結果をディクショナリに変換
            result = {
                "predictions": predictions_original.tolist(),
                "setting": setting,
                "inference_time_ms": inference_time_ms  # 推論時間（ミリ秒）
            }
            
            return jsonify(result)
        else:
            # 従来の方法（データセットからテストデータを読み込む）
            try:
                metrics, pred_data = exp.test(setting, test=1, return_data=True)
                
                # GPU キャッシュをクリア
                torch.cuda.empty_cache()
                
                # 終了時間を記録し、経過時間を計算（ミリ秒単位）
                end_time = time.time()
                inference_time_ms = (end_time - start_time) * 1000
                print(f"推論時間: {inference_time_ms:.2f}ms")
                
                # 結果をディクショナリに変換
                if pred_data is not None:
                    # NumPy配列をリストに変換
                    pred_data = pred_data.tolist() if hasattr(pred_data, 'tolist') else str(pred_data)
                
                result = {
                    "metrics": metrics,
                    "predictions": pred_data,
                    "setting": setting,
                    "inference_time_ms": inference_time_ms  # 推論時間（ミリ秒）
                }
                
                return jsonify(result)
            except Exception as test_error:
                import traceback
                error_trace = traceback.format_exc()
                return jsonify({
                    "error": f"テスト実行中にエラーが発生しました: {str(test_error)}",
                    "traceback": error_trace
                }), 500
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        return jsonify({
            "error": str(e),
            "traceback": error_trace
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """ヘルスチェックエンドポイント"""
    return jsonify({"status": "healthy"})

@app.route('/predict_and_suggest_trade', methods=['POST'])
def predict_and_suggest_trade():
    """予測結果に基づいて取引戦略を提案するAPI"""
    try:
        # リクエストデータの取得
        request_json = request.get_json()
        if not request_json:
            return jsonify({'error': 'リクエストデータが必要です'}), 400

        # 予測の実行（直接predictを呼び出すのではなく、同じロジックを実行）
        # 開始時間を記録
        start_time = time.time()
        
        # リクエストJSONを取得
        if not request_json:
            return jsonify({"error": "リクエストボディが必要です"}), 400
        
        # シード固定
        fix_seed = 2023
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)
        
        # 引数を取得
        args = get_args(request_json)
        
        # 実験クラスを設定
        if args.exp_name == 'partial_train':
            Exp = Exp_Long_Term_Forecast_Partial
        else:
            Exp = Exp_Long_Term_Forecast
        
        # 設定文字列を作成
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            args.class_strategy, ii)
        
        # モデルチェックポイントのパスを探す
        checkpoint_path = f'./checkpoints/{setting}/checkpoint.pth'
        
        # 既定のパスでチェックポイントが見つからない場合、既存のチェックポイントを探す
        if not os.path.exists(checkpoint_path):
            print(f"指定された設定でチェックポイントが見つかりません: {checkpoint_path}")
            
            # ./checkpoints ディレクトリ内のサブディレクトリを探す
            if os.path.exists('./checkpoints'):
                available_checkpoints = [d for d in os.listdir('./checkpoints') 
                                        if os.path.isdir(os.path.join('./checkpoints', d)) and 
                                        os.path.exists(os.path.join('./checkpoints', d, 'checkpoint.pth'))]
                
                # model_idを含むディレクトリを優先して探す
                model_id_checkpoints = [d for d in available_checkpoints if args.model_id in d]
                
                if model_id_checkpoints:
                    # model_idを含むチェックポイントがある場合は最初のものを使用
                    checkpoint_dir = model_id_checkpoints[0]
                    print(f"model_id '{args.model_id}' を含むチェックポイントが見つかりました: {checkpoint_dir}")
                    setting = checkpoint_dir
                    checkpoint_path = f'./checkpoints/{setting}/checkpoint.pth'
                elif available_checkpoints:
                    # model_idを含むものがなければ最初のチェックポイントを使用
                    checkpoint_dir = available_checkpoints[0]
                    print(f"利用可能なチェックポイントが見つかりました: {checkpoint_dir}")
                    setting = checkpoint_dir
                    checkpoint_path = f'./checkpoints/{setting}/checkpoint.pth'
                else:
                    return jsonify({
                        "error": "利用可能なモデルチェックポイントが見つかりません。",
                        "setting": setting
                    }), 404
            else:
                return jsonify({
                    "error": "checkpointsディレクトリが見つかりません。",
                    "setting": setting
                }), 404
        
        print(f"使用するチェックポイント: {checkpoint_path}")
        
        # チェックポイントの設定からモデルパラメータを抽出
        try:
            checkpoint_dir = os.path.basename(os.path.dirname(checkpoint_path))
            print(f"チェックポイントディレクトリ: {checkpoint_dir}")
            
            # モデルパラメータの抽出
            parts = checkpoint_dir.split('_')
            for i, part in enumerate(parts):
                if part == 'dm' and i+1 < len(parts):
                    args.d_model = int(parts[i+1])
                    print(f"チェックポイントから抽出したd_model: {args.d_model}")
                elif part == 'df' and i+1 < len(parts):
                    args.d_ff = int(parts[i+1])
                    print(f"チェックポイントから抽出したd_ff: {args.d_ff}")
                elif part == 'el' and i+1 < len(parts):
                    args.e_layers = int(parts[i+1])
                    print(f"チェックポイントから抽出したe_layers: {args.e_layers}")
                elif part == 'nh' and i+1 < len(parts):
                    args.n_heads = int(parts[i+1])
                    print(f"チェックポイントから抽出したn_heads: {args.n_heads}")
        except Exception as e:
            print(f"チェックポイント名からのパラメータ抽出中にエラー: {e}")
            print("抽出に失敗しても処理を続行します")
        
        # 実験を設定
        exp = Exp(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        
        # 入力データがリクエストに含まれている場合
        input_data = request_json.get('input_data')
        if input_data is not None:
            # 入力データをnumpyに変換
            input_data = np.array(input_data)
            print(f"Input data shape: {input_data.shape}")
            
            # 入力データがスケーリングされていないか確認
            is_scaled = request_json.get('is_scaled', False)
            
            # スケーラーを取得（先に取得してCustomDataProviderに渡す）
            scaler = None
            base_path = os.path.join(os.path.dirname(__file__), 'scripts/multivariate_forecasting/USDJPY/pretreatment/pretreatment_data')
            scaler_joblib_path = os.path.join(base_path, 'usdjpy_scaler.joblib')
            npz_path = os.path.join(base_path, 'usdjpy_windows.npz')
            
            print(f"スケーラー検索パス: {base_path}")
            print(f"joblib スケーラーパス存在: {os.path.exists(scaler_joblib_path)}")
            print(f"npz ファイルパス存在: {os.path.exists(npz_path)}")
            
            # 1. まず、joblibで保存されたスケーラーを探す
            if os.path.exists(scaler_joblib_path):
                try:
                    scaler = joblib.load(scaler_joblib_path)
                    print("joblibスケーラーファイルを読み込みました")
                except Exception as e:
                    print(f"joblibスケーラーファイルの読み込みに失敗しました: {e}")
                    scaler = None
            
            # 2. joblibスケーラーが見つからない場合は、npzから再構築を試みる
            if scaler is None and os.path.exists(npz_path):
                try:
                    npz_data = np.load(npz_path)
                    scaler_min = npz_data['scaler_min']
                    scaler_max = npz_data['scaler_max']
                    scaler = MinMaxScaler()
                    scaler.data_min_ = scaler_min
                    scaler.data_max_ = scaler_max
                    scaler.scale_ = 1.0 / (scaler_max - scaler_min)
                    print("npzファイルからスケーラー情報を再構築しました")
                except Exception as e:
                    print(f"npzファイルからのスケーラー再構築に失敗しました: {e}")
                    scaler = None
            
            # データがスケーリングされていない場合、スケーリングを行う
            if not is_scaled and scaler is not None:
                print("入力データがスケーリングされていないため、スケーリングを行います")
                
                # スケーラーが見つかった場合、入力データをスケーリング
                # 入力データの形状を保存
                original_shape = input_data.shape
                
                # データを2Dに変形 (samples, features)
                reshaped_data = input_data.reshape(-1, original_shape[-1])
                
                # スケーリング実行
                scaled_data = scaler.transform(reshaped_data)
                
                # 元の形状に戻す
                input_data = scaled_data.reshape(original_shape)
                print("入力データをスケーリングしました")
            elif not is_scaled:
                print("スケーラーが見つからないため、スケーリングなしで実行します")
            
            # カスタムデータローダーを作成 (external_scalerを明示的に渡す)
            data_provider = CustomDataProvider(
                input_data=input_data,
                seq_len=args.seq_len,
                pred_len=args.pred_len,
                enc_in=args.enc_in,
                scale=True,  # スケーリング設定を有効化
                external_scaler=scaler  # 外部スケーラーを渡す
            )
            
            # 推論実行
            predictions = []
            predictions_original = []  # 元のスケールに戻した結果を保持
            try:
                print(f"モデルロード中: {checkpoint_path}")
                # strictをFalseに設定して、モデル構造が完全に一致しなくても読み込めるようにする
                model_state = torch.load(checkpoint_path, map_location=torch.device('cpu' if not args.use_gpu else 'cuda'))
                
                # モデル構造の詳細をログに出力（デバッグ用）
                print(f"モデルの状態キー: {list(model_state.keys())[:5]} ...")
                
                # 不一致を許容してロード
                exp.model.load_state_dict(model_state, strict=False)
                print("モデルのロードに成功しました（strict=False）")
                exp.model.eval()
                
                with torch.no_grad():
                    for batch_x, batch_y, batch_x_mark, batch_y_mark in data_provider:
                        batch_x = batch_x.float().to(exp.device)
                        batch_y = batch_y.float().to(exp.device)
                        
                        if 'PEMS' in args.data or 'Solar' in args.data:
                            batch_x_mark = None
                            batch_y_mark = None
                        else:
                            batch_x_mark = batch_x_mark.float().to(exp.device)
                            batch_y_mark = batch_y_mark.float().to(exp.device)
                        
                        # decoder input
                        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(exp.device)
                        
                        print(f"モデル入力形状: batch_x={batch_x.shape}, batch_x_mark={batch_x_mark.shape if batch_x_mark is not None else None}")
                        print(f"デコーダ入力形状: dec_inp={dec_inp.shape}, batch_y_mark={batch_y_mark.shape if batch_y_mark is not None else None}")
                        
                        # encoder - decoder
                        if args.use_amp:
                            with torch.cuda.amp.autocast():
                                if args.output_attention:
                                    outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                else:
                                    outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            if args.output_attention:
                                outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        
                        f_dim = -1 if args.features == 'MS' else 0
                        outputs = outputs[:, -args.pred_len:, f_dim:]
                        outputs = outputs.detach().cpu().numpy()
                        
                        # スケーリングされた結果を保存
                        predictions.append(outputs.copy())
                        
                        # 元のスケールに戻す（ユーザー指定に従う）
                        if args.inverse and hasattr(data_provider, 'inverse_transform'):
                            # スケーラーの状態を確認
                            if data_provider.scaler is None:
                                print("警告: データプロバイダのスケーラーがNoneです、逆変換できません")
                            else:
                                print(f"データプロバイダのスケーラー状態: {type(data_provider.scaler)}")
                            
                            outputs_original = data_provider.inverse_transform(outputs, feature_idx=args.feature_idx)
                            predictions_original.append(outputs_original)
                            print("予測結果を元のスケールに戻しました")
                        else:
                            # スケーラーがない場合や逆変換が無効な場合は、元の結果をそのまま使用
                            predictions_original.append(outputs)
                            print("予測結果はスケーリングされたままです")
            except Exception as model_error:
                import traceback
                error_trace = traceback.format_exc()
                return jsonify({
                    "error": f"モデル推論中にエラーが発生しました: {str(model_error)}",
                    "traceback": error_trace
                }), 500
            
            # 予測結果を結合
            predictions = np.concatenate(predictions, axis=0)
            predictions_original = np.concatenate(predictions_original, axis=0)
            
            # スケール前後の値の範囲をログに出力（デバッグ用）
            if len(predictions.shape) == 3 and predictions.shape[2] > args.feature_idx:
                # マルチ特徴量の場合
                print(f"スケール後の予測値範囲(Close価格): {np.min(predictions[0, :, args.feature_idx]):.6f}～{np.max(predictions[0, :, args.feature_idx]):.6f}")
                print(f"元のスケールに戻した予測値範囲: {np.min(predictions_original):.6f}～{np.max(predictions_original):.6f}")
            else:
                # 単一特徴量の場合
                print(f"スケール後の予測値範囲: {np.min(predictions):.6f}～{np.max(predictions):.6f}")
                print(f"元のスケールに戻した予測値範囲: {np.min(predictions_original):.6f}～{np.max(predictions_original):.6f}")
            
            # GPU キャッシュをクリア
            torch.cuda.empty_cache()
            
            # 終了時間を記録し、経過時間を計算（ミリ秒単位）
            end_time = time.time()
            inference_time_ms = (end_time - start_time) * 1000
            print(f"推論時間: {inference_time_ms:.2f}ms")
            
            # 予測結果をディクショナリに変換
            prediction_result = {
                "predictions": predictions_original.tolist(),  # 元のスケールに戻した予測結果を返す
                "setting": setting,
                "inference_time_ms": inference_time_ms  # 推論時間（ミリ秒）
            }
            
            # 取引戦略の決定（元のスケールに戻した予測結果を使用）
            trade_suggestion = analyze_trade_opportunity(predictions_original)
            
            # レスポンスの作成
            response = {
                'prediction': prediction_result,
                'trade_suggestion': trade_suggestion
            }

            print(response)
            
            return jsonify(response)
        else:
            return jsonify({"error": "入力データが必要です"}), 400
    
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def analyze_trade_opportunity(predictions):
    """予測結果から取引機会を分析する
    
    Args:
        predictions (numpy.ndarray): 予測結果 [batch, pred_len, features] または [batch, pred_len]
                                    ※元のスケールに戻した値であることを想定
        
    Returns:
        dict: 取引提案の詳細
    """
    try:
        # 予測値の形状に応じて処理を分岐
        if len(predictions.shape) == 3:
            # マルチ特徴量の場合、Close価格を使用
            predicted_prices = predictions[0, :, 3]  # Close価格のインデックスは3
        else:
            # 単一特徴量の場合
            predicted_prices = predictions[0]
        
        print(f"取引分析に使用する価格データの範囲: {np.min(predicted_prices):.4f}～{np.max(predicted_prices):.4f}")
        
        # トレンド分析
        price_changes = np.diff(predicted_prices)
        trend_strength = np.mean(price_changes)
        trend_consistency = np.sum(np.sign(price_changes)) / len(price_changes)
        
        # 価格変動の大きさ
        price_volatility = np.std(predicted_prices)
        total_price_change = predicted_prices[-1] - predicted_prices[0]
        max_price_change = np.max(predicted_prices) - np.min(predicted_prices)
        
        # 取引シグナルの生成
        signal = {
            'action': 'HOLD',  # デフォルトはホールド
            'confidence': 0.0,
            'entry_price': float(predicted_prices[0]),
            'target_price': None,
            'stop_loss': None,
            'analysis': {
                'trend_strength': float(trend_strength),
                'trend_consistency': float(trend_consistency),
                'volatility': float(price_volatility),
                'total_change': float(total_price_change),
                'max_change': float(max_price_change)
            }
        }
        
        # トレンドの強さと一貫性に基づく取引判断
        TREND_THRESHOLD = 0.0001  # トレンドの強さの閾値
        CONSISTENCY_THRESHOLD = 0.6  # トレンドの一貫性の閾値
        CONFIDENCE_VOLATILITY_FACTOR = 0.7  # ボラティリティによる信頼度調整係数
        
        if abs(trend_strength) > TREND_THRESHOLD and abs(trend_consistency) > CONSISTENCY_THRESHOLD:
            if trend_strength > 0:
                signal['action'] = 'BUY'
                signal['target_price'] = float(predicted_prices[-1])
                signal['stop_loss'] = float(predicted_prices[0] - max_price_change * 0.5)
            else:
                signal['action'] = 'SELL'
                signal['target_price'] = float(predicted_prices[-1])
                signal['stop_loss'] = float(predicted_prices[0] + max_price_change * 0.5)
            
            # 信頼度の計算
            trend_confidence = min(abs(trend_consistency), 1.0)
            volatility_factor = 1.0 - min(price_volatility * CONFIDENCE_VOLATILITY_FACTOR, 0.5)
            signal['confidence'] = float(trend_confidence * volatility_factor)
        
        # リスク管理の提案
        if signal['action'] in ['BUY', 'SELL']:
            signal['risk_management'] = {
                'position_size_suggestion': '資金の2%以下',
                'max_loss_percentage': '0.5%',
                'trailing_stop': True,
                'split_entry': len(predicted_prices) > 5  # 長期予測の場合は分割エントリーを提案
            }
        
        return signal
        
    except Exception as e:
        print(f"取引分析中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return {
            'action': 'ERROR',
            'error': str(e)
        }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=False) 