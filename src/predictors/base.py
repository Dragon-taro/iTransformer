import os
import json
import yaml
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
from sklearn.preprocessing import MinMaxScaler
import joblib

class BasePredictor(ABC):
    """
    予測モデルの抽象基底クラス。
    全てのモデル固有の予測クラスはこのクラスを継承する必要があります。
    """
    
    def __init__(self, model_id: str, config_path: Optional[str] = None):
        """
        初期化
        
        Args:
            model_id: モデルの識別子
            config_path: 設定ファイルのパス (None の場合はデフォルトパスを使用)
        """
        self.model_id = model_id
        
        # 設定ファイルのパスが指定されていない場合はデフォルトのパスを使用
        if config_path is None:
            config_path = os.path.join('configs', 'models', f'{model_id}.yaml')
        
        # 設定ファイルの読み込み
        self.config = self._load_config(config_path)
        
        # モデル、スケーラー、その他の属性を初期化
        self.model = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() and 
                                  self.config.get('use_gpu', True) else 'cpu')
        
        # スケーラーの読み込み
        self._load_scaler()
        
        # モデルの読み込み
        self.load_model()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        設定ファイルを読み込む
        
        Args:
            config_path: 設定ファイルのパス
            
        Returns:
            設定の辞書
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise ValueError(f"設定ファイルの読み込みに失敗しました: {e}")
    
    def _load_scaler(self) -> None:
        """
        スケーラーを読み込む。設定ファイルで指定されたパスから読み込む。
        """
        scaler_path = self.config.get('scaler_path')
        if not scaler_path:
            return
        
        try:
            # joblibで保存されたスケーラーを読み込む
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print(f"スケーラーを読み込みました: {scaler_path}")
            
            # スケーラーが見つからない場合、npzファイルからの再構築を試みる
            if self.scaler is None:
                npz_path = self.config.get('npz_path')
                if npz_path and os.path.exists(npz_path):
                    npz_data = np.load(npz_path)
                    self.scaler = MinMaxScaler()
                    self.scaler.data_min_ = npz_data['scaler_min']
                    self.scaler.data_max_ = npz_data['scaler_max']
                    self.scaler.scale_ = 1.0 / (self.scaler.data_max_ - self.scaler.data_min_)
                    print(f"npzファイルからスケーラーを再構築しました: {npz_path}")
        except Exception as e:
            print(f"スケーラーの読み込みに失敗しました: {e}")
            self.scaler = None
    
    @abstractmethod
    def load_model(self) -> None:
        """
        モデルをロードする抽象メソッド。
        各サブクラスで実装する必要があります。
        """
        pass
    
    @abstractmethod
    def prepare_input(self, raw_input: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        入力データを前処理する抽象メソッド。
        
        Args:
            raw_input: 生の入力データ (通常はJSONから変換されたディクショナリ)
            
        Returns:
            モデル入力用のテンソルを含む辞書
        """
        pass
    
    @abstractmethod
    def predict(self, inputs: Dict[str, torch.Tensor]) -> np.ndarray:
        """
        モデルで予測を実行する抽象メソッド。
        
        Args:
            inputs: モデル入力用のテンソルを含む辞書
            
        Returns:
            予測結果のnumpy配列
        """
        pass
    
    @abstractmethod
    def postprocess(self, predictions: np.ndarray, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        予測結果を後処理する抽象メソッド。
        
        Args:
            predictions: 予測結果のnumpy配列
            raw_input: 元の入力データ
            
        Returns:
            後処理された予測結果を含む辞書
        """
        pass
    
    def run(self, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        推論パイプラインを実行する
        
        Args:
            raw_input: 生の入力データ
            
        Returns:
            後処理された予測結果
        """
        try:
            # 入力データの準備
            inputs = self.prepare_input(raw_input)
            
            # 予測の実行
            predictions = self.predict(inputs)
            
            # 後処理
            result = self.postprocess(predictions, raw_input)
            
            # 結果にモデルIDと推論時間を追加
            result['model_id'] = self.model_id
            
            return result
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            return {
                "error": str(e),
                "traceback": error_trace,
                "model_id": self.model_id
            }
    
    def inverse_transform(self, predictions: np.ndarray, feature_idx: int = 3) -> np.ndarray:
        """
        予測結果を元のスケールに戻す。
        
        Args:
            predictions: 予測結果のnumpy配列
            feature_idx: 特徴量のインデックス (デフォルトは3=Close価格)
            
        Returns:
            元のスケールに戻した予測結果
        """
        if self.scaler is None:
            print("スケーラーが設定されていないため、逆変換は行わずそのまま返します")
            return predictions
        
        try:
            # 予測結果の次元数に応じて処理を分岐
            if len(predictions.shape) == 3:
                # [batch, time, features] の場合
                # 指定された特徴量のみを抽出
                predictions_feature = predictions[:, :, feature_idx] if predictions.shape[2] > feature_idx else predictions[:, :, 0]
                
                # ダミーデータを作成して逆変換
                dummy = np.zeros((predictions_feature.shape[0] * predictions_feature.shape[1], len(self.scaler.data_min_)))
                dummy_idx = 0
                
                for i in range(predictions_feature.shape[0]):
                    for j in range(predictions_feature.shape[1]):
                        dummy[dummy_idx, feature_idx] = predictions_feature[i, j]
                        dummy_idx += 1
                
                # 逆変換
                dummy_inverted = self.scaler.inverse_transform(dummy)
                
                # 元の形状に戻す
                result = np.zeros_like(predictions_feature)
                dummy_idx = 0
                for i in range(predictions_feature.shape[0]):
                    for j in range(predictions_feature.shape[1]):
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