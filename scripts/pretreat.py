#!/usr/bin/env python
"""
iTransformer 前処理実行スクリプト
モデル別の前処理を設定ファイルに基づいて実行します
"""
import sys
from pathlib import Path
import importlib

# パスを追加して相対インポートを使えるようにする
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    """前処理モジュールのmain関数を実行"""
    try:
        from scripts.preprocessing.run import main as preprocess_main
        preprocess_main()
    except ImportError as e:
        print(f"エラー: 前処理モジュールのインポートに失敗しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 