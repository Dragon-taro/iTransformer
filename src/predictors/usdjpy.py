import os
import numpy as np
import torch
from typing import Dict, Any, Optional

from .default import DefaultPredictor

class USDJPYPredictor(DefaultPredictor):
    """
    USDJPY向けのカスタムプレディクタ実装。
    """
    
    def __init__(self, model_id: str = 'usdjpy', config_path: Optional[str] = None):
        """
        初期化
        
        Args:
            model_id: モデルの識別子 (デフォルトは 'usdjpy')
            config_path: 設定ファイルのパス (None の場合はデフォルトパスを使用)
        """
        # デフォルトのスケーラーパスを設定
        self.default_scaler_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'scripts/multivariate_forecasting/USDJPY/pretreatment/pretreatment_data/usdjpy_scaler.joblib'
        )
        
        self.default_npz_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'scripts/multivariate_forecasting/USDJPY/pretreatment/pretreatment_data/usdjpy_windows.npz'
        )
        
        # 親クラスの初期化
        super().__init__(model_id, config_path)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        設定ファイルを読み込み、USDJPY固有のデフォルト値を設定する
        
        Args:
            config_path: 設定ファイルのパス
            
        Returns:
            設定の辞書
        """
        # 親クラスの設定読み込み
        config = super()._load_config(config_path)
        
        # USDJPY固有の設定をデフォルト値として追加
        usdjpy_defaults = {
            'model_type': 'iTransformer',
            'data_type': 'custom',
            'features': 'M',
            'seq_len': 60,  # 過去60分
            'label_len': 10,  # ラベル長10分
            'pred_len': 10,  # 将来10分を予測
            'enc_in': 4,   # OHLC=4つの特徴量
            'dec_in': 4,   # デコーダ入力も同じく4特徴量
            'c_out': 1,    # 出力はClose価格1つ
            'd_model': 128,
            'n_heads': 8,
            'e_layers': 2,
            'd_layers': 1,
            'd_ff': 128,
            'factor': 1,
            'embed': 'timeF',
            'distil': True,
            'include_trade_suggestion': True,  # 取引シグナルを生成
            'feature_idx': 3,  # Close価格は3番目
            'scaler_path': self.default_scaler_path,
            'npz_path': self.default_npz_path,
            'inverse': True  # デフォルトで逆変換を有効化
        }
        
        # デフォルト値を設定ファイルの値で上書き
        for key, value in usdjpy_defaults.items():
            if key not in config:
                config[key] = value
        
        return config
    
    def postprocess(self, predictions: np.ndarray, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        予測結果を後処理する。
        USDJPY向けのカスタム後処理を実装する。
        
        Args:
            predictions: 予測結果のnumpy配列
            raw_input: 元の入力データ
            
        Returns:
            後処理された予測結果を含む辞書
        """
        # 親クラスの後処理を呼び出す
        result = super().postprocess(predictions, raw_input)
        
        # USDJPY固有の後処理
        # 例：取引コスト（スプレッド）を考慮したシグナル調整
        if 'trade_suggestion' in result:
            trade = result['trade_suggestion']
            
            # USDJPYの典型的なスプレッドを考慮（約0.2-0.3銭）
            spread = 0.03  # 3 pips
            
            # スプレッドを考慮した実現可能な利益を計算
            if trade['action'] == 'BUY':
                entry_with_spread = trade['entry_price'] + spread/2
                potential_profit = trade['target_price'] - entry_with_spread
                # スプレッドを考慮しても利益が見込める場合のみシグナルを維持
                if potential_profit <= spread:
                    trade['action'] = 'HOLD'
                    trade['adjusted_reason'] = 'スプレッドを考慮すると取引コストが利益を上回るため'
                else:
                    trade['entry_price'] = float(entry_with_spread)
                    trade['spread_adjusted'] = True
            
            elif trade['action'] == 'SELL':
                entry_with_spread = trade['entry_price'] - spread/2
                potential_profit = entry_with_spread - trade['target_price']
                if potential_profit <= spread:
                    trade['action'] = 'HOLD'
                    trade['adjusted_reason'] = 'スプレッドを考慮すると取引コストが利益を上回るため'
                else:
                    trade['entry_price'] = float(entry_with_spread)
                    trade['spread_adjusted'] = True
            
            result['trade_suggestion'] = trade
            
            # 最適な時間枠の提案を追加
            result['trade_suggestion']['recommended_timeframe'] = self._get_recommended_timeframe(predictions)
        
        return result
    
    def _get_recommended_timeframe(self, predictions: np.ndarray) -> str:
        """
        予測結果に基づいて推奨される取引時間枠を返す
        
        Args:
            predictions: 予測結果のnumpy配列
            
        Returns:
            推奨時間枠の文字列
        """
        # 予測値の形状に応じて処理を分岐
        if len(predictions.shape) == 3:
            # マルチ特徴量の場合、Close価格を使用
            feature_idx = self.config.get('feature_idx', 3)
            predicted_prices = predictions[0, :, feature_idx] if predictions.shape[2] > feature_idx else predictions[0, :, 0]
        else:
            # 単一特徴量の場合
            predicted_prices = predictions[0]
        
        # ボラティリティ計算
        volatility = np.std(predicted_prices)
        price_range = np.max(predicted_prices) - np.min(predicted_prices)
        trend_strength = abs(predicted_prices[-1] - predicted_prices[0])
        
        # 分析に基づいて時間枠を提案
        if volatility > 0.15 and price_range > 0.5:
            return 'M5 (5分足) - 高ボラティリティに対応するためのスキャルピング'
        elif trend_strength > 0.3:
            return 'M15 (15分足) - 明確な短期トレンドを追随'
        else:
            return 'M30 (30分足) または H1 (1時間足) - 安定した値動きを利用'
        
        return 'M15 (15分足)'  # デフォルト推奨 