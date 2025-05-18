import os
import numpy as np
import torch
from typing import Dict, Any, Optional, List

from .base import BasePredictor
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
from experiments.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial
import time

class DefaultPredictor(BasePredictor):
    """
    デフォルトのプレディクタ実装。
    モデル固有の実装がない場合に使用される一般的なiTransformerモデル用のプレディクタ。
    """
    
    def load_model(self) -> None:
        """
        設定ファイルに基づいてiTransformerモデルをロードする。
        """
        try:
            # チェックポイントのパスを取得
            checkpoint_path = self.config.get('checkpoint_path')
            if not checkpoint_path:
                # チェックポイントパスが設定されていない場合は、設定情報からパスを生成
                model_id = self.model_id
                model_type = self.config.get('model_type', 'iTransformer')
                data_type = self.config.get('data_type', 'custom')
                features = self.config.get('features', 'M')
                seq_len = self.config.get('seq_len', 60)
                label_len = self.config.get('label_len', 48)
                pred_len = self.config.get('pred_len', 10)
                d_model = self.config.get('d_model', 128)
                n_heads = self.config.get('n_heads', 8)
                e_layers = self.config.get('e_layers', 2)
                d_layers = self.config.get('d_layers', 1)
                d_ff = self.config.get('d_ff', 128)
                factor = self.config.get('factor', 1)
                embed = self.config.get('embed', 'timeF')
                distil = self.config.get('distil', True)
                des = self.config.get('des', 'test')
                class_strategy = self.config.get('class_strategy', 'projection')
                
                # 設定文字列を生成
                setting = f'{model_id}_{model_type}_{data_type}_{features}_ft{seq_len}_sl{label_len}_pl{pred_len}_dm{d_model}_nh{n_heads}_el{e_layers}_dl{d_layers}_df{d_ff}_fc{factor}_eb{embed}_dt{distil}_{des}_{class_strategy}_0'
                
                # チェックポイントのパスを生成
                checkpoint_path = os.path.join('checkpoints', setting, 'checkpoint.pth')
                
                # チェックポイントが存在するか確認
                if not os.path.exists(checkpoint_path):
                    print(f"指定された設定でチェックポイントが見つかりません: {checkpoint_path}")
                    # 既存のチェックポイントを探す
                    if os.path.exists('./checkpoints'):
                        available_checkpoints = [d for d in os.listdir('./checkpoints') 
                                            if os.path.isdir(os.path.join('./checkpoints', d)) and 
                                            os.path.exists(os.path.join('./checkpoints', d, 'checkpoint.pth'))]
                        
                        # model_idを含むディレクトリを優先して探す
                        model_id_checkpoints = [d for d in available_checkpoints if model_id in d]
                        
                        if model_id_checkpoints:
                            # model_idを含むチェックポイントがある場合は最初のものを使用
                            checkpoint_dir = model_id_checkpoints[0]
                            print(f"model_id '{model_id}' を含むチェックポイントが見つかりました: {checkpoint_dir}")
                            checkpoint_path = os.path.join('./checkpoints', checkpoint_dir, 'checkpoint.pth')
                        elif available_checkpoints:
                            # model_idを含むものがなければ最初のチェックポイントを使用
                            checkpoint_dir = available_checkpoints[0]
                            print(f"利用可能なチェックポイントが見つかりました: {checkpoint_dir}")
                            checkpoint_path = os.path.join('./checkpoints', checkpoint_dir, 'checkpoint.pth')
                        else:
                            raise FileNotFoundError("利用可能なモデルチェックポイントが見つかりません。")
            
            print(f"使用するチェックポイント: {checkpoint_path}")
            
            # 実験クラスを設定
            exp_name = self.config.get('exp_name', 'MTSF')
            if exp_name == 'partial_train':
                exp_class = Exp_Long_Term_Forecast_Partial
            else:
                exp_class = Exp_Long_Term_Forecast
            
            # Argを作成
            args = self._create_args_from_config()
            
            # 実験を設定
            self.exp = exp_class(args)
            
            # モデルをロード
            model_state = torch.load(checkpoint_path, map_location=self.device)
            self.exp.model.load_state_dict(model_state, strict=False)
            self.exp.model.eval()
            
            print(f"モデルのロードが完了しました: {self.model_id}")
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"モデルのロード中にエラーが発生しました: {e}")
            print(error_trace)
            raise
    
    def _create_args_from_config(self):
        """
        設定ファイルからargsオブジェクトを作成する。
        """
        import argparse
        
        args = argparse.Namespace()
        
        # 必須パラメータ
        args.is_training = 0  # 推論モード
        args.model_id = self.config.get('model_id', self.model_id)
        args.model = self.config.get('model_type', 'iTransformer')
        args.data = self.config.get('data_type', 'custom')
        
        # データロード関連
        args.root_path = self.config.get('root_path', './data/electricity/')
        args.data_path = self.config.get('data_path', 'electricity.csv')
        args.features = self.config.get('features', 'M')
        args.target = self.config.get('target', 'Close')
        args.freq = self.config.get('freq', 'h')
        args.checkpoints = self.config.get('checkpoints', './checkpoints/')
        
        # 予測タスク
        args.seq_len = self.config.get('seq_len', 60)
        args.label_len = self.config.get('label_len', 48)
        args.pred_len = self.config.get('pred_len', 10)
        
        # モデル定義
        args.enc_in = self.config.get('enc_in', 4)  # 入力特徴量の数（デフォルトはOHLC=4）
        args.dec_in = self.config.get('dec_in', 4)
        args.c_out = self.config.get('c_out', 1)
        args.d_model = self.config.get('d_model', 128)
        args.n_heads = self.config.get('n_heads', 8)
        args.e_layers = self.config.get('e_layers', 2)
        args.d_layers = self.config.get('d_layers', 1)
        args.d_ff = self.config.get('d_ff', 128)
        args.moving_avg = self.config.get('moving_avg', 25)
        args.factor = self.config.get('factor', 1)
        args.distil = self.config.get('distil', True)
        args.dropout = self.config.get('dropout', 0.1)
        args.embed = self.config.get('embed', 'timeF')
        args.activation = self.config.get('activation', 'gelu')
        args.output_attention = self.config.get('output_attention', False)
        args.do_predict = self.config.get('do_predict', False)
        
        # 最適化
        args.num_workers = self.config.get('num_workers', 10)
        args.itr = self.config.get('itr', 1)
        args.train_epochs = self.config.get('train_epochs', 10)
        args.batch_size = self.config.get('batch_size', 32)
        args.patience = self.config.get('patience', 3)
        args.learning_rate = self.config.get('learning_rate', 0.0001)
        args.des = self.config.get('des', 'test')
        args.loss = self.config.get('loss', 'MSE')
        args.lradj = self.config.get('lradj', 'type1')
        args.use_amp = self.config.get('use_amp', False)
        
        # GPU
        args.use_gpu = torch.cuda.is_available() and self.config.get('use_gpu', True)
        args.gpu = self.config.get('gpu', 0)
        args.use_multi_gpu = self.config.get('use_multi_gpu', False)
        args.devices = self.config.get('devices', '0,1,2,3')
        
        # iTransformer
        args.exp_name = self.config.get('exp_name', 'MTSF')
        args.channel_independence = self.config.get('channel_independence', False)
        args.inverse = self.config.get('inverse', True)  # デフォルトで逆変換を有効化
        args.class_strategy = self.config.get('class_strategy', 'projection')
        args.target_root_path = self.config.get('target_root_path', './data/electricity/')
        args.target_data_path = self.config.get('target_data_path', 'electricity.csv')
        args.efficient_training = self.config.get('efficient_training', False)
        args.use_norm = self.config.get('use_norm', True)
        args.partial_start_index = self.config.get('partial_start_index', 0)
        
        # スケーリング設定
        args.scale = self.config.get('scale', True)
        args.feature_idx = self.config.get('feature_idx', 3)  # 3=Close
        
        # GPUの設定
        if args.use_gpu and args.use_multi_gpu:
            args.devices = args.devices.replace(' ', '')
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]
        
        return args
    
    def prepare_input(self, raw_input: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        入力データを前処理する。
        
        Args:
            raw_input: 生の入力データ
            
        Returns:
            モデル入力用のテンソルを含む辞書
        """
        # 開始時間を記録
        start_time = time.time()
        
        # 入力データを取得
        input_data = np.array(raw_input['input_data'])
        
        batch_size, seq_len, n_features = input_data.shape
        pred_len = self.config.get('pred_len', 10)
        
        # 時間特徴量の次元数
        timestamp_feat_dim = 4  # 位置エンコーディングの次元数
        
        # 入力用の位置エンコーディング [batch, seq_len, feat_dim]
        batch_x_mark = np.zeros((batch_size, seq_len, timestamp_feat_dim))
        for i in range(batch_size):
            for j in range(seq_len):
                # 単純な位置エンコーディング
                batch_x_mark[i, j, 0] = j / seq_len  # 正規化された位置
                batch_x_mark[i, j, 1] = np.sin(j * (2 * np.pi / seq_len))  # sin位置
                batch_x_mark[i, j, 2] = np.cos(j * (2 * np.pi / seq_len))  # cos位置
                batch_x_mark[i, j, 3] = (j % 7) / 7.0  # 週ごとのパターン
        
        # 出力用の位置エンコーディング [batch, pred_len, feat_dim]
        batch_y_mark = np.zeros((batch_size, pred_len, timestamp_feat_dim))
        for i in range(batch_size):
            for j in range(pred_len):
                # 予測部分の位置エンコーディング
                pos = seq_len + j
                batch_y_mark[i, j, 0] = pos / (seq_len + pred_len)  # 正規化された位置
                batch_y_mark[i, j, 1] = np.sin(pos * (2 * np.pi / (seq_len + pred_len)))
                batch_y_mark[i, j, 2] = np.cos(pos * (2 * np.pi / (seq_len + pred_len)))
                batch_y_mark[i, j, 3] = (pos % 7) / 7.0  # 週ごとのパターン
        
        # ダミー出力データ
        dummy_output = np.zeros((batch_size, pred_len, n_features))
        
        # テンソルに変換
        batch_x = torch.FloatTensor(input_data).to(self.device)
        batch_y = torch.FloatTensor(dummy_output).to(self.device)
        batch_x_mark = torch.FloatTensor(batch_x_mark).to(self.device)
        batch_y_mark = torch.FloatTensor(batch_y_mark).to(self.device)
        
        # decoder input
        label_len = self.config.get('label_len', 48)
        dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(self.device)
        
        # 処理時間を計算
        prep_time = time.time() - start_time
        
        return {
            'batch_x': batch_x,
            'batch_y': batch_y,
            'batch_x_mark': batch_x_mark,
            'batch_y_mark': batch_y_mark,
            'dec_inp': dec_inp,
            'prep_time': prep_time
        }
    
    def predict(self, inputs: Dict[str, torch.Tensor]) -> np.ndarray:
        """
        モデルで予測を実行する。
        
        Args:
            inputs: モデル入力用のテンソルを含む辞書
            
        Returns:
            予測結果のnumpy配列
        """
        start_time = time.time()
        
        batch_x = inputs['batch_x']
        batch_y = inputs['batch_y']
        batch_x_mark = inputs['batch_x_mark']
        batch_y_mark = inputs['batch_y_mark']
        dec_inp = inputs['dec_inp']
        
        with torch.no_grad():
            # モデル設定の取得
            args = self.exp.args
            use_amp = args.use_amp
            output_attention = args.output_attention
            features = args.features
            pred_len = args.pred_len
            
            # encoder - decoder
            if use_amp:
                with torch.cuda.amp.autocast():
                    if output_attention:
                        outputs = self.exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if output_attention:
                    outputs = self.exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            f_dim = -1 if features == 'MS' else 0
            outputs = outputs[:, -pred_len:, f_dim:]
            predictions = outputs.detach().cpu().numpy()
        
        # 推論時間を計算
        inference_time = time.time() - start_time
        
        # 元のデータの形状と予測の形状をログに出力（デバッグ用）
        print(f"入力形状: {batch_x.shape}, 予測形状: {predictions.shape}")
        print(f"推論時間: {inference_time * 1000:.2f}ms")
        
        return predictions
    
    def postprocess(self, predictions: np.ndarray, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        予測結果を後処理する。
        
        Args:
            predictions: 予測結果のnumpy配列
            raw_input: 元の入力データ
            
        Returns:
            後処理された予測結果を含む辞書
        """
        start_time = time.time()
        
        # 設定からインバース変換の有無を取得
        inverse = self.config.get('inverse', True)
        feature_idx = self.config.get('feature_idx', 3)  # 3=Close
        
        # 元のスケールに戻す
        if inverse and self.scaler is not None:
            predictions_original = self.inverse_transform(predictions, feature_idx=feature_idx)
            print("予測結果を元のスケールに戻しました")
        else:
            predictions_original = predictions
            print("元のスケールへの変換をスキップしました")
        
        # 処理時間を計算
        postproc_time = time.time() - start_time
        
        # 結果をディクショナリに変換
        result = {
            "predictions": predictions_original.tolist(),
            "inference_time_ms": postproc_time * 1000  # 後処理時間（ミリ秒）
        }
        
        # 取引戦略の分析（設定で有効になっている場合）
        if self.config.get('include_trade_suggestion', False):
            result['trade_suggestion'] = self._analyze_trade_opportunity(predictions_original)
        
        return result
    
    def _analyze_trade_opportunity(self, predictions: np.ndarray) -> Dict[str, Any]:
        """
        予測結果から取引機会を分析する。
        
        Args:
            predictions: 予測結果 (元のスケールに戻したもの)
            
        Returns:
            取引提案の詳細
        """
        try:
            # 予測値の形状に応じて処理を分岐
            if len(predictions.shape) == 3:
                # マルチ特徴量の場合、Close価格を使用
                feature_idx = self.config.get('feature_idx', 3)  # 3=Close
                predicted_prices = predictions[0, :, feature_idx] if predictions.shape[2] > feature_idx else predictions[0, :, 0]
            else:
                # 単一特徴量の場合
                predicted_prices = predictions[0]
            
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