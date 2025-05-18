import os
import importlib
from typing import Dict, Type, Any, Optional

from .base import BasePredictor

# プレディクタインスタンスのキャッシュ
_PREDICTOR_CACHE: Dict[str, BasePredictor] = {}

# プレディクタクラスのキャッシュ
_PREDICTOR_CLASS_CACHE: Dict[str, Type[BasePredictor]] = {}

def get_predictor_class(model_id: str) -> Type[BasePredictor]:
    """
    モデルIDに対応するプレディクタクラスを取得する。
    
    Args:
        model_id: モデルの識別子
        
    Returns:
        対応するBasePredictor派生クラス
    
    Raises:
        ImportError: プレディクタクラスが見つからない場合
    """
    # キャッシュに存在する場合はそれを返す
    if model_id in _PREDICTOR_CLASS_CACHE:
        return _PREDICTOR_CLASS_CACHE[model_id]
    
    # モジュール名を生成 (例: usdjpy → src.predictors.usdjpy)
    module_name = f"src.predictors.{model_id}"
    
    try:
        # モジュールをインポート
        module = importlib.import_module(module_name)
        
        # クラス名を推測 (例: usdjpy → USDJPYPredictor)
        class_name = f"{model_id.title().replace('_', '')}Predictor"
        
        # モジュールからクラスを取得
        predictor_class = getattr(module, class_name)
        
        # BasePredictor のサブクラスであることを確認
        if not issubclass(predictor_class, BasePredictor):
            raise ImportError(f"{class_name} は BasePredictor のサブクラスではありません")
        
        # キャッシュに保存
        _PREDICTOR_CLASS_CACHE[model_id] = predictor_class
        
        return predictor_class
        
    except (ImportError, AttributeError) as e:
        # モデル固有の実装が見つからない場合はデフォルトプレディクタを使用
        if os.path.exists(os.path.join("configs", "models", f"{model_id}.yaml")):
            from .default import DefaultPredictor
            _PREDICTOR_CLASS_CACHE[model_id] = DefaultPredictor
            return DefaultPredictor
        
        # それでも見つからない場合はエラー
        raise ImportError(f"モデル '{model_id}' のプレディクタクラスが見つかりません: {e}")

def get_predictor(model_id: str, config_path: Optional[str] = None) -> BasePredictor:
    """
    モデルIDに対応するプレディクタインスタンスを取得する。
    
    Args:
        model_id: モデルの識別子
        config_path: 設定ファイルのパス (None の場合はデフォルトパスを使用)
        
    Returns:
        BasePredictor インスタンス
    """
    # 設定ファイルが指定されている場合は毎回新しいインスタンスを作成
    if config_path is not None:
        predictor_class = get_predictor_class(model_id)
        return predictor_class(model_id, config_path)
    
    # キャッシュに存在する場合はそれを返す
    cache_key = model_id
    if cache_key in _PREDICTOR_CACHE:
        return _PREDICTOR_CACHE[cache_key]
    
    # 新しいインスタンスを作成してキャッシュに保存
    predictor_class = get_predictor_class(model_id)
    predictor = predictor_class(model_id)
    _PREDICTOR_CACHE[cache_key] = predictor
    
    return predictor

def clear_predictor_cache() -> None:
    """
    プレディクタキャッシュをクリアする。
    """
    _PREDICTOR_CACHE.clear() 