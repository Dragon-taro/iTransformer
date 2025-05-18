import pandas as pd
import numpy as np
import os
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_yaml_config(config_path: str) -> Dict:
    """
    YAML設定ファイルをロード
    
    Args:
        config_path: 設定ファイルパス
        
    Returns:
        設定辞書
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml_config(config: Dict, output_path: str) -> None:
    """
    設定をYAMLファイルとして保存
    
    Args:
        config: 設定辞書
        output_path: 出力パス
    """
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def load_preprocessed_data(npz_path: str) -> Dict[str, np.ndarray]:
    """
    前処理済みデータをロード
    
    Args:
        npz_path: npzファイルパス
        
    Returns:
        データ辞書
    """
    data = np.load(npz_path)
    return {key: data[key] for key in data.files}


def load_scaler(scaler_path: str) -> Any:
    """
    保存されたスケーラーをロード
    
    Args:
        scaler_path: スケーラーファイルパス
        
    Returns:
        スケーラーオブジェクト
    """
    return joblib.load(scaler_path)


def create_scaler_from_params(scaler_min: np.ndarray, scaler_max: np.ndarray) -> MinMaxScaler:
    """
    保存されたパラメータからMinMaxScalerを再構築
    
    Args:
        scaler_min: data_min_値
        scaler_max: data_max_値
        
    Returns:
        再構築されたMinMaxScaler
    """
    scaler = MinMaxScaler()
    scaler.data_min_ = scaler_min
    scaler.data_max_ = scaler_max
    scaler.scale_ = 1.0 / (scaler_max - scaler_min)
    scaler.min_ = 0 - scaler_min * scaler.scale_
    return scaler


def load_csv_with_dates(csv_path: str) -> pd.DataFrame:
    """
    日付付きCSVをロード
    
    Args:
        csv_path: CSVファイルパス
        
    Returns:
        DataFrame
    """
    df = pd.read_csv(csv_path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    return df


def get_available_models(config_dir: str = 'configs/models') -> List[str]:
    """
    利用可能なモデルIDのリストを取得
    
    Args:
        config_dir: モデル設定ディレクトリ
        
    Returns:
        モデルIDのリスト
    """
    config_path = Path(config_dir)
    if not config_path.exists():
        return []
    
    return [f.stem for f in config_path.glob('*.yaml')]


def preprocess_and_save_model_registry(preprocessor_class, data_path: str, 
                                       config_path: str, output_dir: str, model_id: str) -> Dict:
    """
    データを前処理してモデルレジストリに必要な情報を保存
    
    Args:
        preprocessor_class: 前処理クラス
        data_path: データファイルパス
        config_path: 設定ファイルパス
        output_dir: 出力ディレクトリ
        model_id: モデルID
        
    Returns:
        レジストリ情報辞書
    """
    # 前処理実行
    preprocessor = preprocessor_class(config_path=config_path)
    if output_dir:
        preprocessor.output_dir = Path(output_dir)
        preprocessor.output_dir.mkdir(parents=True, exist_ok=True)
    
    data_dict = preprocessor.preprocess(data_path, model_id=model_id)
    
    # モデルレジストリ情報を作成
    registry_info = {
        'model_id': model_id,
        'data': {
            'npz_path': str(preprocessor.output_dir / f"{model_id}_windows.npz"),
            'scaler_path': str(preprocessor.output_dir / f"{model_id}_scaler.joblib"),
            'train_csv': str(preprocessor.output_dir / f"{model_id}_X_train_wdate.csv"),
            'val_csv': str(preprocessor.output_dir / f"{model_id}_X_val_wdate.csv"),
            'test_csv': str(preprocessor.output_dir / f"{model_id}_X_test_wdate.csv")
        },
        'config': preprocessor.config
    }
    
    # レジストリ情報を保存
    registry_path = Path(output_dir) / f"{model_id}_registry.yaml"
    save_yaml_config(registry_info, str(registry_path))
    
    return registry_info 