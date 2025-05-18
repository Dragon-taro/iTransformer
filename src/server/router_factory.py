from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
import numpy as np
import logging
import time
import json
from datetime import datetime

# 型ヒントのためのモジュール import（実際の実装はPredictorFactoryで行う）
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.predictors.base import BasePredictor

logger = logging.getLogger("iTransformer-server")

class PredictionInput(BaseModel):
    """予測リクエストの共通入力スキーマ"""
    data: List[List[float]] = Field(..., description="入力時系列データ。形状は[時間ステップ数, 特徴量数]")
    
    class Config:
        schema_extra = {
            "example": {
                "data": [[101.23, 101.45, 101.20, 101.30], [101.30, 101.50, 101.25, 101.40]]
            }
        }

class PredictionResponse(BaseModel):
    """予測レスポンスの共通スキーマ"""
    model_id: str
    predictions: List[List[float]]
    processed_at: str
    processing_time_ms: float
    metadata: Optional[Dict[str, Any]] = None

def get_predictor_factory():
    """Predictor Factoryを遅延インポート（循環参照を避けるため）"""
    from src.predictors.factory import get_predictor
    return get_predictor

def create_router(model_id: str, config: Dict[str, Any]) -> APIRouter:
    """指定されたモデルIDとその設定に基づいてルーターを作成する
    
    Args:
        model_id: モデルの固有識別子
        config: モデルの設定情報
        
    Returns:
        設定されたFastAPIルーター
    """
    router = APIRouter()
    get_predictor = get_predictor_factory()
    
    @router.get("/")
    async def model_info():
        """モデル情報を返すエンドポイント"""
        # 機密情報や内部パラメータを除外した公開用設定
        public_config = {
            "model_id": model_id,
            "description": config.get("description", ""),
            "version": config.get("version", "1.0.0"),
            "features": config.get("features", []),
            "target": config.get("target", []),
            "last_updated": config.get("last_updated", "")
        }
        return public_config
    
    @router.post("/predict", response_model=PredictionResponse)
    async def predict(payload: PredictionInput = Body(...)):
        """予測を実行するエンドポイント"""
        start_time = time.time()
        
        try:
            # Predictor Factoryからモデルを取得
            predictor = get_predictor(model_id)

            # 入力データを準備
            # dataを単一バッチとして3次元配列に変換 [batch, seq_len, n_features]
            prepared_data = {
                'input_data': np.array([payload.data])  # バッチ次元を追加
            }
            
            # 入力データを前処理
            prepared_input = predictor.prepare_input(prepared_data)
            
            # 予測を実行
            predictions = predictor.predict(prepared_input)
            
            # 後処理（必要に応じて）
            processed_results = predictor.postprocess(predictions, prepared_data)
            
            # レスポンスを構築
            response = {
                "model_id": model_id,
                "predictions": processed_results.get("predictions", []),
                "processed_at": datetime.now().isoformat(),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "metadata": {
                    "model_version": config.get("version", "1.0.0"),
                    "window_size": config.get("seq_len", config.get("window_size")),
                    "pred_len": config.get("pred_len", 10),
                    "features_used": config.get("features", "M")
                }
            }
            
            # 取引提案が含まれている場合は追加
            if "trade_suggestion" in processed_results:
                response["trade_suggestion"] = processed_results["trade_suggestion"]
            
            return response
            
        except Exception as e:
            logger.error(f"予測処理中にエラーが発生しました: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"予測の実行中にエラーが発生しました: {str(e)}"
            )
    
    @router.post("/batch_predict")
    async def batch_predict(payloads: List[PredictionInput] = Body(...)):
        """複数データセットの一括予測を行うエンドポイント"""
        start_time = time.time()
        results = []
        
        try:
            predictor = get_predictor(model_id)
            
            for i, payload in enumerate(payloads):
                item_start = time.time()
                
                # データを3次元配列に変換
                prepared_data = {
                    'input_data': np.array([payload.data])  # バッチ次元を追加
                }
                
                # 個別予測と同様の処理
                prepared_input = predictor.prepare_input(prepared_data)
                predictions = predictor.predict(prepared_input)
                processed_results = predictor.postprocess(predictions, prepared_data)
                
                results.append({
                    "batch_index": i,
                    "predictions": processed_results.get("predictions", []),
                    "processing_time_ms": (time.time() - item_start) * 1000,
                    "trade_suggestion": processed_results.get("trade_suggestion", None)
                })
            
            return {
                "model_id": model_id,
                "batch_size": len(payloads),
                "results": results,
                "total_processing_time_ms": (time.time() - start_time) * 1000
            }
            
        except Exception as e:
            logger.error(f"バッチ予測中にエラーが発生しました: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"バッチ予測の実行中にエラーが発生しました: {str(e)}"
            )
    
    return router 