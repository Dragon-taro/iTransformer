#!/usr/bin/env python
import argparse
import os
from pathlib import Path
import importlib
import sys

from .base import BasePreprocessor
from .utils import load_yaml_config, save_yaml_config, preprocess_and_save_model_registry


def main():
    parser = argparse.ArgumentParser(description="iTransformer前処理実行ツール")
    parser.add_argument("--config", type=str, required=True, help="設定YAMLファイルへのパス")
    parser.add_argument("--data_path", type=str, help="処理するCSVファイルへのパス")
    parser.add_argument("--data_dir", type=str, help="処理するCSVファイルを含むディレクトリ")
    parser.add_argument("--output_dir", type=str, help="出力ディレクトリ")
    parser.add_argument("--model_id", type=str, help="モデルID")
    parser.add_argument("--processor", type=str, default="USJPYPreprocessor", 
                        help="使用する前処理クラス (例: USJPYPreprocessor)")
    args = parser.parse_args()
    
    # 設定ファイルの確認
    if not os.path.exists(args.config):
        print(f"エラー: 設定ファイル {args.config} が見つかりません")
        sys.exit(1)
    
    # データパスの確認
    if not args.data_path and not args.data_dir:
        print("エラー: --data_path または --data_dir を指定してください")
        sys.exit(1)
    
    if args.data_path and not os.path.exists(args.data_path):
        print(f"エラー: データファイル {args.data_path} が見つかりません")
        sys.exit(1)
    
    if args.data_dir and not os.path.isdir(args.data_dir):
        print(f"エラー: データディレクトリ {args.data_dir} が見つかりません")
        sys.exit(1)
    
    # 前処理クラスを動的にインポート
    try:
        # まず自身のモジュールから探す
        processor_class = getattr(importlib.import_module("scripts.preprocessing"), args.processor)
    except (ImportError, AttributeError):
        try:
            # 次に直接指定されたモジュールパスから探す
            module_name, class_name = args.processor.rsplit(".", 1)
            module = importlib.import_module(module_name)
            processor_class = getattr(module, class_name)
        except (ValueError, ImportError, AttributeError):
            print(f"エラー: 前処理クラス '{args.processor}' が見つかりません")
            sys.exit(1)
    
    # 出力ディレクトリ設定
    output_dir = args.output_dir
    if not output_dir:
        # 設定ファイルから取得
        config = load_yaml_config(args.config)
        output_dir = config.get('output_dir', 'data/processed')
    
    # 出力ディレクトリの作成
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # モデルID設定
    model_id = args.model_id
    if not model_id:
        # データパスからファイル名を抽出
        if args.data_path:
            model_id = Path(args.data_path).stem
        else:
            model_id = Path(args.data_dir).name
    
    # 前処理の実行
    try:
        if args.data_path:
            # 単一ファイル処理
            registry_info = preprocess_and_save_model_registry(
                processor_class, 
                args.data_path, 
                args.config, 
                output_dir, 
                model_id
            )
            print(f"前処理成功: {model_id}")
            print(f"レジストリ情報: {registry_info['data']['npz_path']}")
        
        elif args.data_dir:
            # ディレクトリ内の全ファイル処理
            preprocessor = processor_class(config_path=args.config)
            preprocessor.output_dir = Path(output_dir)
            preprocessor.run_from_directory(args.data_dir, output_dir)
            print(f"ディレクトリ処理成功: {args.data_dir}")
    
    except Exception as e:
        print(f"前処理中にエラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 