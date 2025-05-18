#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import argparse
import json
from pprint import pprint

# srcディレクトリの親ディレクトリ（プロジェクトルート）をパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# 絶対インポートを使用
from predictors import get_predictor

def load_sample_data(data_path=None, seq_len=60, n_features=4):
    """
    サンプルデータを読み込む関数

    Args:
        data_path: データファイルのパス、Noneの場合はランダムに生成
        seq_len: シーケンス長
        n_features: 特徴量の数（OHLC=4）

    Returns:
        numpy.ndarray: シェイプ [1, seq_len, n_features] のサンプルデータ
    """
    if data_path and os.path.exists(data_path):
        try:
            # CSVからの読み込み
            df = pd.read_csv(data_path)
            if "date" in df.columns:
                df = df.drop(columns=["date"])
            
            # データ形状の確認
            if len(df) < seq_len:
                raise ValueError(f"データサイズが足りません: {len(df)} < {seq_len}")
            
            # 最初のseq_len行を使用
            data = df.iloc[:seq_len].values
            
            # 特徴量の数が不足している場合は0で埋める
            if data.shape[1] < n_features:
                padded_data = np.zeros((seq_len, n_features))
                padded_data[:, :data.shape[1]] = data
                data = padded_data
            
            # バッチ次元を追加
            return data.reshape(1, seq_len, n_features)
        
        except Exception as e:
            print(f"データ読み込みエラー: {e}")
            print("ランダムデータを生成します")
    
    # ランダムデータを生成
    print(f"ランダムデータを生成: [1, {seq_len}, {n_features}]")
    return np.random.rand(1, seq_len, n_features)

def main():
    parser = argparse.ArgumentParser(description="プレディクタテスト")
    parser.add_argument("--model_id", type=str, default="usdjpy", help="モデルID")
    parser.add_argument("--data_path", type=str, default=None, help="入力データのCSVパス（指定しない場合はランダムデータ）")
    parser.add_argument("--seq_len", type=int, default=60, help="シーケンス長")
    parser.add_argument("--n_features", type=int, default=4, help="特徴量数")
    parser.add_argument("--output", type=str, default=None, help="出力JSONファイルパス")
    parser.add_argument("--config", type=str, default=None, help="設定ファイルのパス")
    args = parser.parse_args()

    try:
        # データ読み込み
        input_data = load_sample_data(args.data_path, args.seq_len, args.n_features)
        print(f"入力データ形状: {input_data.shape}")
        
        # プレディクタ初期化
        print(f"モデル '{args.model_id}' のプレディクタを初期化")
        predictor = get_predictor(args.model_id, args.config)
        
        # 推論実行
        print("推論実行中...")
        raw_input = {"input_data": input_data.tolist()}
        result = predictor.run(raw_input)
        
        # エラーチェック
        if "error" in result:
            print(f"エラー: {result['error']}")
            if "traceback" in result:
                print(result["traceback"])
            return 1
        
        # 結果表示
        print("\n予測結果:")
        if "predictions" in result:
            predictions = np.array(result["predictions"])
            print(f"予測形状: {predictions.shape}")
            print(f"予測値（最初の数値）: {predictions.flatten()[:5]}...")
        
        # 取引提案の表示
        if "trade_suggestion" in result:
            print("\n取引提案:")
            trade = result["trade_suggestion"]
            print(f"アクション: {trade.get('action', 'N/A')}")
            print(f"信頼度: {trade.get('confidence', 'N/A')}")
            print(f"エントリー価格: {trade.get('entry_price', 'N/A')}")
            print(f"目標価格: {trade.get('target_price', 'N/A')}")
            
            if "recommended_timeframe" in trade:
                print(f"推奨時間枠: {trade['recommended_timeframe']}")
        
        # 結果をファイルに保存
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n結果を {args.output} に保存しました")
        
        return 0
    
    except Exception as e:
        import traceback
        print(f"エラー: {e}")
        print(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 