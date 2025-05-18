import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from pathlib import Path
import joblib
from datetime import datetime, timedelta

logger = logging.getLogger("iTransformer-server")

class DataProvider:
    """
    予測に必要なデータを取得・前処理するプロバイダクラス
    
    このクラスは以下の責務を持ちます:
    1. 必要なスケーラーのロード
    2. 特徴量の正規化/非正規化
    3. 入力データの整形とwindow切り出し
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        データプロバイダの初期化
        
        Args:
            config: モデル設定（スケーラーパス、ウィンドウサイズなどの情報を含む）
        """
        self.config = config
        self.window_size = config.get("window_size", 96)  # デフォルト: 96タイムステップ
        self.pred_len = config.get("pred_len", 24)  # デフォルト: 24タイムステップ先を予測
        self.features = config.get("features", [])
        self.scaler = None
        
        # スケーラーのロード（存在する場合）
        scaler_path = config.get("scaler_path", None)
        if scaler_path:
            self._load_scaler(scaler_path)
    
    def _load_scaler(self, scaler_path: str):
        """
        指定されたパスからスケーラーをロードする
        
        Args:
            scaler_path: スケーラーが保存されたパス
        """
        try:
            full_path = Path(scaler_path)
            if not full_path.exists():
                logger.warning(f"スケーラーが見つかりません: {scaler_path}")
                return
                
            self.scaler = joblib.load(full_path)
            logger.info(f"スケーラーをロードしました: {scaler_path}")
        except Exception as e:
            logger.error(f"スケーラーのロード中にエラーが発生しました: {e}")
            self.scaler = None
    
    def prepare_data(self, data: Union[List[List[float]], np.ndarray]) -> np.ndarray:
        """
        入力データを予測用に準備する
        
        Args:
            data: 生の入力データ配列
            
        Returns:
            モデル入力用に整形されたNumPy配列
        """
        # リストからNumPy配列に変換
        if isinstance(data, list):
            data = np.array(data, dtype=np.float32)
            
        # 必要に応じて特徴量の次元を確認・調整
        if len(data.shape) == 1:
            # 一次元の場合、二次元に変換（1つの特徴量として扱う）
            data = data.reshape(-1, 1)
        
        # 特徴量数が設定と一致するか確認
        expected_features = len(self.features) if self.features else None
        if expected_features and data.shape[1] != expected_features:
            logger.warning(
                f"入力特徴量数({data.shape[1]})が期待される特徴量数({expected_features})と一致しません"
            )
        
        # スケーリング処理（スケーラーが存在する場合）
        if self.scaler:
            try:
                data = self.scaler.transform(data)
            except Exception as e:
                logger.error(f"スケーリング中にエラーが発生しました: {e}")
        
        # ウィンドウサイズの確認と切り出し
        if data.shape[0] < self.window_size:
            # 入力データが足りない場合はパディング
            padding = np.zeros((self.window_size - data.shape[0], data.shape[1]), dtype=np.float32)
            data = np.vstack([padding, data])
            logger.warning(f"入力データが短すぎるためパディングしました。元のサイズ: {data.shape[0] - padding.shape[0]}")
        
        # 最新のウィンドウを抽出
        if data.shape[0] > self.window_size:
            data = data[-self.window_size:]
            
        # バッチ次元を追加 [time, features] -> [batch=1, time, features]
        data = np.expand_dims(data, axis=0)
        
        return data
    
    def inverse_transform(self, predictions: np.ndarray, target_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        予測結果を元のスケールに戻す
        
        Args:
            predictions: モデルからの予測結果
            target_indices: 対象特徴量のインデックス
            
        Returns:
            元のスケールに戻された予測値
        """
        if self.scaler is None:
            # スケーラーがない場合はそのまま返す
            return predictions
        
        # バッチ次元を除去する場合
        if len(predictions.shape) == 3 and predictions.shape[0] == 1:
            predictions = predictions.squeeze(0)
        
        try:
            # sklearn系のスケーラーは全特徴量を必要とするため、ダミーデータを作成
            dummy_data = np.zeros((predictions.shape[0], len(self.features) if self.features else self.scaler.n_features_in_))
            
            # ターゲット特徴量のみが予測される場合
            if target_indices is not None:
                for i, idx in enumerate(target_indices):
                    dummy_data[:, idx] = predictions[:, i]
            else:
                # 出力が全特徴量を含む場合
                if predictions.shape[1] == dummy_data.shape[1]:
                    dummy_data = predictions
                else:
                    # 出力の特徴量数が異なる場合（例：最初の特徴量のみを予測）
                    dummy_data[:, 0:predictions.shape[1]] = predictions
            
            # 逆変換
            inversed = self.scaler.inverse_transform(dummy_data)
            
            # ターゲット特徴量のみを返す
            if target_indices is not None:
                return inversed[:, target_indices]
            else:
                if predictions.shape[1] < inversed.shape[1]:
                    return inversed[:, 0:predictions.shape[1]]
                return inversed
                
        except Exception as e:
            logger.error(f"逆変換中にエラーが発生しました: {e}")
            return predictions
    
    def get_timestamps(self, length: int, freq: str = '1H', end_time: Optional[datetime] = None) -> List[str]:
        """
        予測用のタイムスタンプ配列を生成
        
        Args:
            length: 生成するタイムスタンプの数
            freq: タイムスタンプの頻度（pandas頻度文字列）
            end_time: 最終時刻（Noneの場合は現在時刻）
            
        Returns:
            ISO形式のタイムスタンプリスト
        """
        if end_time is None:
            end_time = datetime.now()
        
        # 最終時刻から予測期間分のタイムスタンプを生成
        timestamps = pd.date_range(
            end=end_time, 
            periods=length+1,  # 現在+予測期間
            freq=freq
        )[1:]  # 最初（現在）を除外
        
        return timestamps.strftime('%Y-%m-%dT%H:%M:%SZ').tolist() 