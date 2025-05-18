from fastapi import APIRouter, HTTPException, Body, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import json

from .utils import get_model_registry, check_model_health

logger = logging.getLogger("iTransformer-server")

# 共通のレスポンススキーマ
class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None

# モデル管理用のルーター
def create_admin_router() -> APIRouter:
    """管理用APIルーターを作成"""
    router = APIRouter(prefix="/admin", tags=["admin"])
    
    @router.get("/models", response_model=ApiResponse)
    async def list_models():
        """登録されているモデル一覧を取得"""
        try:
            registry = get_model_registry()
            
            # 機密情報を除外したパブリック情報を返す
            public_info = {}
            for model_id, config in registry.items():
                public_info[model_id] = {
                    "description": config.get("description", ""),
                    "version": config.get("version", "1.0.0"),
                    "features": config.get("features", []),
                    "last_updated": config.get("last_updated", ""),
                    "health_status": "unknown"  # 後で更新
                }
                
                # モデルの健全性をチェック
                checkpoint_path = config.get("checkpoint_path")
                if checkpoint_path:
                    public_info[model_id]["health_status"] = "healthy" if check_model_health(checkpoint_path) else "unhealthy"
            
            return {
                "success": True,
                "message": f"{len(registry)}個のモデルが見つかりました",
                "data": public_info
            }
        except Exception as e:
            logger.error(f"モデル一覧の取得中にエラーが発生しました: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"モデル情報の取得に失敗しました: {str(e)}"
            )
    
    @router.get("/health", response_model=ApiResponse)
    async def server_health():
        """サーバーの健全性チェック"""
        try:
            # レジストリとモデルの健全性を確認
            registry = get_model_registry()
            unhealthy_models = []
            
            for model_id, config in registry.items():
                checkpoint_path = config.get("checkpoint_path")
                if checkpoint_path and not check_model_health(checkpoint_path):
                    unhealthy_models.append(model_id)
            
            # メモリ使用量など追加のシステム情報を収集（実装例）
            import psutil
            system_info = {
                "memory_usage_percent": psutil.virtual_memory().percent,
                "cpu_usage_percent": psutil.cpu_percent(),
                "disk_usage_percent": psutil.disk_usage('/').percent
            }
            
            return {
                "success": True,
                "message": "サーバーは正常に動作しています" if not unhealthy_models else f"{len(unhealthy_models)}個の異常モデルがあります",
                "data": {
                    "total_models": len(registry),
                    "healthy_models": len(registry) - len(unhealthy_models),
                    "unhealthy_models": unhealthy_models,
                    "system_info": system_info
                }
            }
        except ImportError:
            # psutilがない場合の簡易チェック
            return {
                "success": True,
                "message": "基本的な健全性チェックは正常です (詳細な情報にはpsutilが必要です)",
                "data": {
                    "total_models": len(get_model_registry())
                }
            }
        except Exception as e:
            logger.error(f"健全性チェック中にエラーが発生しました: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"健全性チェックに失敗しました: {str(e)}"
            )
    
    return router

# WebSocket用の拡張（もし実装するなら）
def create_realtime_router() -> APIRouter:
    """リアルタイムデータ用のルーター"""
    from fastapi import WebSocket, WebSocketDisconnect
    
    router = APIRouter(prefix="/realtime", tags=["realtime"])
    
    # 接続中のクライアント管理
    class ConnectionManager:
        def __init__(self):
            self.active_connections: Dict[str, List[WebSocket]] = {}
            
        async def connect(self, websocket: WebSocket, model_id: str):
            await websocket.accept()
            if model_id not in self.active_connections:
                self.active_connections[model_id] = []
            self.active_connections[model_id].append(websocket)
            
        def disconnect(self, websocket: WebSocket, model_id: str):
            if model_id in self.active_connections:
                self.active_connections[model_id].remove(websocket)
                
        async def send_update(self, message: Dict[str, Any], model_id: str):
            if model_id in self.active_connections:
                for connection in self.active_connections[model_id]:
                    await connection.send_json(message)
    
    manager = ConnectionManager()
    
    @router.websocket("/{model_id}/ws")
    async def websocket_endpoint(websocket: WebSocket, model_id: str):
        """WebSocketエンドポイント - リアルタイム予測用"""
        try:
            await manager.connect(websocket, model_id)
            logger.info(f"新しいWebSocket接続: {model_id}")
            
            # 接続確認メッセージ
            await websocket.send_json({
                "status": "connected",
                "model_id": model_id,
                "message": f"モデル {model_id} へ接続しました"
            })
            
            try:
                while True:
                    # クライアントからのメッセージを待機
                    data = await websocket.receive_json()
                    
                    # 予測処理用のコードをここに実装（将来的に）
                    # from src.predictors.factory import get_predictor
                    # predictor = get_predictor(model_id)
                    # prediction = predictor.predict(data)
                    
                    # サンプルレスポンス
                    await websocket.send_json({
                        "status": "prediction",
                        "timestamp": "2023-01-01T12:00:00Z",
                        "data": {
                            "message": "リアルタイム予測機能は開発中です"
                        }
                    })
                    
            except WebSocketDisconnect:
                manager.disconnect(websocket, model_id)
                logger.info(f"WebSocket切断: {model_id}")
                
        except Exception as e:
            logger.error(f"WebSocket処理中にエラーが発生しました: {e}")
            await websocket.close(code=1011, reason=f"エラー: {str(e)}")
    
    return router

# gRPC拡張のためのプレースホルダー（将来的な実装用）
def setup_grpc_server():
    """
    gRPCサーバー設定のプレースホルダー
    
    将来的にgRPCサポートが必要になった場合に実装
    """
    pass 