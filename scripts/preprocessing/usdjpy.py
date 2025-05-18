from .base import BasePreprocessor
import os
from pathlib import Path
from typing import Dict, Any


class USJPYPreprocessor(BasePreprocessor):
    """USDJPY通貨ペア用の前処理クラス"""
    
    def __init__(self, config_path: str = None, config: Dict = None):
        """
        初期化
        
        Args:
            config_path: 設定ファイルへのパス
            config: 直接設定を渡す辞書
        """
        # デフォルト設定
        default_config = {
            'window_size': 60,
            'target_size': 10,
            'frequency': '1T',
            'price_cols': ['Open', 'High', 'Low', 'Close'],
            'target_col': 'Close',
            'datetime_col': 'Gmt time',
            'datetime_format': '%d.%m.%Y %H:%M:%S.%f'
        }
        
        # config_pathかconfigが渡されていればそれを使い、そうでなければデフォルト設定を使用
        if not config_path and not config:
            config = default_config
        
        super().__init__(config_path, config)
    
    def preprocess(self, data_path: str, model_id: str = "usdjpy") -> Dict[str, Any]:
        """
        USJPYデータの前処理を実行
        
        Args:
            data_path: データファイルへのパス
            model_id: モデルID (デフォルト: "usdjpy")
            
        Returns:
            処理されたデータ構造
        """
        print(f"データファイルを読み込み中: {data_path}")
        
        # データロード
        df = self.load_data(data_path)
        
        # 欠損値処理
        df = self.handle_missing_values(df)
        
        # 正規化
        scaled_data, cols_to_use = self.normalize_data(df)
        
        # スライディングウィンドウ作成
        X, y = self.create_windows(scaled_data, cols_to_use)
        
        # データ分割
        window_dates = df.index[:len(X)]
        data_dict = self.split_data(X, y, window_dates)
        
        # データ保存
        self.save_data(data_dict, cols_to_use, model_id)
        
        # 結果表示
        print(f"トレーニングセット: {data_dict['X_train'].shape}, {data_dict['y_train'].shape}")
        print(f"検証セット: {data_dict['X_val'].shape}, {data_dict['y_val'].shape}")
        print(f"テストセット: {data_dict['X_test'].shape}, {data_dict['y_test'].shape}")
        
        return data_dict
    
    def run_from_directory(self, data_dir: str, output_dir: str = None, pattern: str = "*.csv") -> None:
        """
        ディレクトリ内の全CSVファイルを処理
        
        Args:
            data_dir: データディレクトリ
            output_dir: 出力ディレクトリ (指定された場合は設定を上書き)
            pattern: 処理対象ファイルのパターン
        """
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        data_path = Path(data_dir)
        csv_files = list(data_path.glob(pattern))
        
        if not csv_files:
            print(f"警告: {data_dir}内に{pattern}に一致するファイルが見つかりません")
            return
        
        for csv_file in csv_files:
            # ファイル名からmodel_idを生成 (拡張子を除外)
            model_id = f"usdjpy_{csv_file.stem}"
            self.preprocess(str(csv_file), model_id)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="USDJPY通貨ペアの前処理")
    parser.add_argument("--config", type=str, help="設定YAMLファイルへのパス")
    parser.add_argument("--data_path", type=str, help="処理するCSVファイルへのパス")
    parser.add_argument("--data_dir", type=str, help="処理するCSVファイルを含むディレクトリ")
    parser.add_argument("--output_dir", type=str, help="出力ディレクトリ")
    parser.add_argument("--model_id", type=str, default="usdjpy", help="モデルID")
    args = parser.parse_args()
    
    preprocessor = USJPYPreprocessor(config_path=args.config)
    
    if args.output_dir:
        preprocessor.output_dir = Path(args.output_dir)
        preprocessor.output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.data_path:
        # 単一ファイル処理
        preprocessor.preprocess(args.data_path, model_id=args.model_id)
    elif args.data_dir:
        # ディレクトリ内の全ファイル処理
        preprocessor.run_from_directory(args.data_dir, args.output_dir)
    else:
        print("エラー: --data_path または --data_dir を指定してください") 