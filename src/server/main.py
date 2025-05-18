from fastapi import FastAPI, HTTPException
from pathlib import Path
import yaml
import logging
import uvicorn
from typing import Dict, Any

from .router_factory import create_router
from .extensions import create_admin_router, create_realtime_router
from .utils import setup_logging

# ロガーの設定
setup_logging(log_level="INFO", log_file="logs/server.log")
logger = logging.getLogger("iTransformer-server")

app = FastAPI(
    title="iTransformer Inference API",
    description="複数の金融時系列予測モデルに対応した推論APIサーバー",
    version="1.0.0",
)

# モデル設定のキャッシュ
model_configs: Dict[str, Dict[str, Any]] = {}

@app.get("/")
async def root():
    """ルートエンドポイント - 利用可能なモデル一覧を返す"""
    return {
        "status": "online",
        "available_models": list(model_configs.keys()),
        "endpoints": [f"/{model_id}/predict" for model_id in model_configs.keys()]
    }

def load_model_configs():
    """モデル設定ファイルを読み込む"""
    config_dir = Path("configs/models")
    if not config_dir.exists():
        logger.warning(f"設定ディレクトリが見つかりません: {config_dir}")
        return {}
    
    configs = {}
    for config_path in config_dir.glob("*.yaml"):
        model_id = config_path.stem
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                configs[model_id] = config
                logger.info(f"モデル設定を読み込みました: {model_id}")
        except Exception as e:
            logger.error(f"モデル設定の読み込みに失敗しました {config_path}: {e}")
    
    return configs

def register_model_routers():
    """モデル設定からルーターを登録"""
    global model_configs
    model_configs = load_model_configs()
    
    if not model_configs:
        logger.warning("有効なモデル設定がありません。エンドポイントは生成されません。")
        return
    
    # 各モデルのルーターを登録
    for model_id, config in model_configs.items():
        try:
            router = create_router(model_id, config)
            app.include_router(
                router, 
                prefix=f"/{model_id}", 
                tags=[model_id]
            )
            logger.info(f"モデルエンドポイントを登録しました: /{model_id}")
        except Exception as e:
            logger.error(f"モデル {model_id} のルーター登録に失敗しました: {e}")

# 起動時にモデルルーターを登録
@app.on_event("startup")
async def startup_event():
    logger.info("サーバー起動中...")
    
    # 基本モデルルーターの登録
    register_model_routers()
    
    # 管理APIルーターの登録
    admin_router = create_admin_router()
    app.include_router(admin_router)
    logger.info("管理APIエンドポイントを登録しました")
    
    # リアルタイムAPIルーターの登録
    realtime_router = create_realtime_router()
    app.include_router(realtime_router)
    logger.info("リアルタイムAPIエンドポイントを登録しました")
    
    logger.info("サーバー起動完了")

if __name__ == "__main__":
    uvicorn.run("src.server.main:app", host="0.0.0.0", port=8000, reload=True) 