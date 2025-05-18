import logging
import yaml
from pathlib import Path
import json
from typing import Dict, Any, List, Optional
import time
import os
import sys
import torch

logger = logging.getLogger("iTransformer-server")

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    ロギングの設定
    
    Args:
        log_level: ログレベル名 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: ログファイルパス (Noneの場合は標準出力のみ)
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )

def load_config_file(config_path: str) -> Dict[str, Any]:
    """
    YAML設定ファイルを読み込む
    
    Args:
        config_path: YAML設定ファイルのパス
        
    Returns:
        設定辞書
        
    Raises:
        FileNotFoundError: 設定ファイルが見つからない場合
        ValueError: 設定ファイルの解析エラー
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise ValueError(f"設定ファイルの読み込みに失敗しました: {e}")

def check_model_health(checkpoint_path: str, device: str = "cpu") -> bool:
    """
    モデルチェックポイントの健全性チェック
    
    Args:
        checkpoint_path: モデルチェックポイントのパス
        device: ロードするデバイス
        
    Returns:
        チェックポイントが有効な場合はTrue、無効ならFalse
    """
    try:
        path = Path(checkpoint_path)
        if not path.exists():
            logger.error(f"チェックポイントが見つかりません: {checkpoint_path}")
            return False
        
        # ファイルサイズのチェック
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb < 0.1:  # 100KB未満の場合は警告
            logger.warning(f"チェックポイントのサイズが小さすぎます: {size_mb:.2f} MB")
        
        # チェックポイントをロードしてみる
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 一般的なチェックポイント構造の確認
        required_keys = ["model_state_dict", "epoch"]
        missing_keys = [key for key in required_keys if key not in checkpoint]
        
        if missing_keys:
            logger.warning(f"チェックポイントに一部キーがありません: {missing_keys}")
            
        return True
    except Exception as e:
        logger.error(f"チェックポイントのチェック中にエラーが発生しました: {e}")
        return False

def get_model_registry() -> Dict[str, Dict[str, Any]]:
    """
    利用可能なモデル設定を取得
    
    Returns:
        モデルID: 設定情報のマップ
    """
    registry = {}
    model_configs_dir = Path("configs/models")
    
    if not model_configs_dir.exists():
        logger.warning(f"モデル設定ディレクトリが見つかりません: {model_configs_dir}")
        return registry
    
    for config_file in model_configs_dir.glob("*.yaml"):
        model_id = config_file.stem
        try:
            config = load_config_file(str(config_file))
            registry[model_id] = config
        except Exception as e:
            logger.error(f"モデル {model_id} の設定読み込みに失敗しました: {e}")
    
    return registry

def record_prediction_metrics(model_id: str, inference_time_ms: float, input_shape: List[int]):
    """
    予測のメトリクスを記録（ログまたはモニタリングシステムへ）
    
    Args:
        model_id: モデルID
        inference_time_ms: 推論時間（ミリ秒）
        input_shape: 入力データ形状
    """
    # シンプルなログ記録（実際の環境では監視システムへ送信するなど）
    logger.info(
        f"Metrics - model: {model_id}, "
        f"inference_time: {inference_time_ms:.2f}ms, "
        f"input_shape: {input_shape}"
    ) 