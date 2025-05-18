import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any


class BasePreprocessor(ABC):
    """基本前処理クラス - 共通のデータ処理ロジックを実装"""
    
    def __init__(self, config_path: str = None, config: Dict = None):
        """
        設定ファイルからパラメータをロード
        
        Args:
            config_path: YAML設定ファイルへのパス
            config: 直接辞書として渡す設定（config_pathより優先）
        """
        self.config = self._load_config(config_path) if config_path else config or {}
        
        # 共通パラメータをロード
        self.window_size = self.config.get('window_size', 60)
        self.target_size = self.config.get('target_size', 10)
        self.frequency = self.config.get('frequency', '1T')
        self.train_ratio = self.config.get('train_ratio', 0.8)
        self.val_ratio = self.config.get('val_ratio', 0.1)
        # test_ratio = 1 - (train_ratio + val_ratio)
        
        self.output_dir = Path(self.config.get('output_dir', 'pretreatment_data'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.price_cols = self.config.get('price_cols', ['Open', 'High', 'Low', 'Close'])
        self.target_col = self.config.get('target_col', 'Close')
        
        # スケーラー
        self.scaler = None
    
    def _load_config(self, config_path: str) -> Dict:
        """YAML設定ファイルをロード"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        CSVファイルからデータをロード
        
        Args:
            filepath: CSVファイルパス
            
        Returns:
            前処理済みDataFrame
        """
        # ファイル拡張子に基づいて読み込み方法を決定
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            raise ValueError(f"未対応のファイル形式: {filepath}")
        
        # 日時カラムを処理
        datetime_col = self.config.get('datetime_col', 'Gmt time')
        datetime_format = self.config.get('datetime_format', '%d.%m.%Y %H:%M:%S.%f')
        
        if datetime_col in df.columns:
            df['Datetime'] = pd.to_datetime(df[datetime_col], format=datetime_format)
            df = df.drop(columns=[datetime_col])
        
        # インデックス設定
        df = df.set_index('Datetime').sort_index()
        
        # リサンプリングと欠損値補完
        df = df.resample(self.frequency).ffill()
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """欠損値処理 - 基本的にffillを使用"""
        strategy = self.config.get('missing_value_strategy', 'ffill')
        
        if strategy == 'ffill':
            return df.fillna(method='ffill')
        elif strategy == 'bfill':
            return df.fillna(method='bfill')
        elif strategy == 'mean':
            return df.fillna(df.mean())
        else:
            return df.fillna(method='ffill')
    
    def normalize_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        データを正規化
        
        Args:
            df: 入力DataFrame
            
        Returns:
            正規化されたNumPy配列
        """
        # 価格カラムが存在するか確認
        cols_to_use = [col for col in self.price_cols if col in df.columns]
        if not cols_to_use:
            cols_to_use = df.columns.drop('Volume') if 'Volume' in df.columns else df.columns
        
        data = df[cols_to_use].values
        
        # スケーラー作成
        self.scaler = MinMaxScaler()
        scaled = self.scaler.fit_transform(data)
        
        return scaled, cols_to_use
    
    def create_windows(self, scaled_data: np.ndarray, cols_to_use: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        スライディングウィンドウを生成
        
        Args:
            scaled_data: 正規化されたデータ
            cols_to_use: 使用するカラム名
            
        Returns:
            X, y: 入力ウィンドウと対応する目標値
        """
        X, y = [], []
        target_idx = cols_to_use.index(self.target_col) if self.target_col in cols_to_use else 0
        
        for i in range(len(scaled_data) - self.window_size - self.target_size + 1):
            X.append(scaled_data[i:i+self.window_size])
            y.append(scaled_data[i+self.window_size:i+self.window_size+self.target_size, target_idx])
        
        return np.array(X), np.array(y)
    
    def split_data(self, X: np.ndarray, y: np.ndarray, dates: pd.DatetimeIndex) -> Dict[str, np.ndarray]:
        """
        データをトレーニング/検証/テストに分割
        
        Args:
            X: 入力ウィンドウ
            y: 目標値
            dates: 日付インデックス
            
        Returns:
            分割されたデータセット
        """
        n = len(X)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        train_dates = dates[:train_end]
        val_dates = dates[train_end:val_end]
        test_dates = dates[val_end:]
        
        return {
            'X_train': X_train, 'y_train': y_train, 'train_dates': train_dates,
            'X_val': X_val, 'y_val': y_val, 'val_dates': val_dates,
            'X_test': X_test, 'y_test': y_test, 'test_dates': test_dates
        }
    
    def save_data(self, data_dict: Dict[str, np.ndarray], cols_to_use: List[str], model_id: str) -> None:
        """
        処理済みデータを保存
        
        Args:
            data_dict: 分割されたデータセット
            cols_to_use: 使用するカラム名
            model_id: モデルID（ファイル名の一部として使用）
        """
        # 1. npzファイルとして保存
        output_path = self.output_dir / f"{model_id}_windows.npz"
        np.savez(
            output_path,
            X_train=data_dict['X_train'], y_train=data_dict['y_train'],
            X_val=data_dict['X_val'], y_val=data_dict['y_val'],
            X_test=data_dict['X_test'], y_test=data_dict['y_test'],
            scaler_min=self.scaler.data_min_, scaler_max=self.scaler.data_max_
        )
        
        # 2. スケーラーをjoblibで保存
        scaler_path = self.output_dir / f"{model_id}_scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        
        # 3. CSVファイルとして書き出し
        self._save_with_dates(data_dict, cols_to_use, model_id)
        
        print(f"Saved preprocessed data to {output_path}, {scaler_path}, and CSV files in {self.output_dir}/")
    
    def _save_with_dates(self, data_dict: Dict[str, np.ndarray], cols_to_use: List[str], model_id: str) -> None:
        """
        データセットを日付付きのCSVとして保存
        
        Args:
            data_dict: 分割されたデータセット
            cols_to_use: 使用するカラム名
            model_id: モデルID（ファイル名の一部として使用）
        """
        for split in ['train', 'val', 'test']:
            # X（入力）データの保存
            X = data_dict[f'X_{split}']
            dates = data_dict[f'{split}_dates']

            # X: (サンプル数, window_size, feature数) → (サンプル数, window_size*feature数)
            X_reshaped = X.reshape(X.shape[0], -1)
            # カラム名: Open×60, High×60, Low×60, Close×60 の順
            X_cols = []
            for col in cols_to_use:
                X_cols += [col] * self.window_size
            df_X = pd.DataFrame(X_reshaped, columns=X_cols)
            df_X.insert(0, 'date', dates.strftime('%Y-%m-%d %H:%M:%S'))
            df_X.to_csv(self.output_dir / f"{model_id}_X_{split}_wdate.csv", index=False)

            # y（目標）データの保存
            y = data_dict[f'y_{split}']
            # yカラム名: Closeをtarget_size回繰り返し
            y_cols = [self.target_col] * self.target_size
            df_y = pd.DataFrame(y, columns=y_cols)
            df_y.insert(0, 'date', dates.strftime('%Y-%m-%d %H:%M:%S'))
            df_y.to_csv(self.output_dir / f"{model_id}_y_{split}_wdate.csv", index=False)
    
    @abstractmethod
    def preprocess(self, data_path: str, model_id: str) -> Dict[str, Any]:
        """
        前処理メインメソッド - 派生クラスでの実装が必要
        
        Args:
            data_path: データファイルへのパス
            model_id: モデルID（ファイル名の一部として使用）
            
        Returns:
            処理されたデータ構造
        """
        pass 