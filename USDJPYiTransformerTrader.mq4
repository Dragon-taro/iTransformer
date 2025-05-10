//+------------------------------------------------------------------+
//|                                     USDJPYiTransformerTrader.mq4 |
//|                                   iTransformerを使用した自動取引EA |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "iTransformer Trader"
#property link      "https://github.com/thuml/iTransformer"
#property version   "1.00"
#property strict

// 取引設定
extern string API_Settings = "======= API設定 =======";
extern string API_URL = "http://localhost/predict_and_suggest_trade";
extern int API_Timeout = 30000; // ミリ秒
extern bool UseSSL = false;

extern string Trading_Settings = "======= 取引設定 =======";
extern int PredictionIntervalMinutes = 60; // 予測を取得する間隔（分）
extern double LotSize = 0.01; // 取引ロットサイズ
extern bool UseFixedLotSize = true; // 固定ロットサイズを使用する
extern double RiskPercent = 2.0; // 資金に対するリスク（%）
extern double MinimumConfidence = 0.6; // 最小信頼度

extern string Position_Settings = "======= ポジション設定 =======";
extern bool UseSplitEntry = true; // 分割エントリーを使用するか
extern int SplitEntryCount = 3; // 分割エントリーの回数
extern bool UseTrailingStop = true; // トレーリングストップを使用するか
extern int TrailingStopPoints = 30; // トレーリングストップのポイント

// グローバル変数
int lastCalculatedBar = 0;
datetime lastApiCallTime = 0;
string lastTradeSignal = "NONE";
double lastSignalConfidence = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    // 通貨ペアがUSDJPYかどうか確認
    if (Symbol() != "USDJPY")
    {
        Print("このEAはUSDJPY専用です。現在のシンボル: ", Symbol());
        return INIT_FAILED;
    }
    
    // 時間足が1分足かどうか確認
    if (Period() != PERIOD_M1)
    {
        Print("このEAは1分足チャート上で動作します。現在の時間足: ", Period());
        return INIT_FAILED;
    }

    Print("USDJPYiTransformerTrader 初期化成功");
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print("USDJPYiTransformerTrader 終了: 理由コード=", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    // 新しいバーが形成された場合、または初回のみ実行
    if (isNewBar() || lastCalculatedBar == 0)
    {
        lastCalculatedBar = Bars;
        
        // 指定した間隔でのみAPIを呼び出す
        if (TimeCurrent() - lastApiCallTime >= PredictionIntervalMinutes * 60)
        {
            lastApiCallTime = TimeCurrent();
            CallPredictionAPI();
        }
    }
}

//+------------------------------------------------------------------+
//| 新しいバーが形成されたかどうかをチェック                           |
//+------------------------------------------------------------------+
bool isNewBar()
{
    if (Bars > lastCalculatedBar)
        return true;
    return false;
}

//+------------------------------------------------------------------+
//| 予測APIを呼び出す                                                |
//+------------------------------------------------------------------+
void CallPredictionAPI()
{
    // 過去60分のOHLCデータを収集
    string inputData = PrepareInputData();
    
    // リクエストデータを作成
    string requestBody = "{\"input_data\":" + inputData + ",\"seq_len\":60,\"pred_len\":10,\"enc_in\":4}";
    
    Print("APIリクエスト送信中: ", API_URL);
    Print("リクエストボディ: ", requestBody);
    
    // char 配列にコピーし、実バイト長で ArrayResize
    char bodyArray[];
    int bodyLen = StringToCharArray(requestBody, bodyArray) - 1; // 末尾のヌル文字を除外
    if(bodyLen > 0) {
        ArrayResize(bodyArray, bodyLen); // 末尾のヌル文字を取り除く
    }
    
    // JSON ヘッダ
    string header = "Content-Type: application/json; charset=utf-8\r\nAccept: application/json";
    
    // APIリクエストを送信
    char responseBody[];
    string responseHeaders;
    int result = WebRequest(
        "POST",
        API_URL,
        header,
        API_Timeout,
        bodyArray,
        responseBody,
        responseHeaders
    );
    
    // レスポンスを処理
    if (result == -1)
    {
        int errorCode = GetLastError();
        Print("API呼び出しエラー: ", errorCode, " - ", GetErrorDescription(errorCode));
        if (errorCode == 4060) 
            Print("WebRequestの許可が必要です。MT4のツール > オプション > 専門家アドバイザー で設定してください。");
        return;
    }
    
    string response = CharArrayToString(responseBody);
    
    // デバッグのためにレスポンスを詳細に出力
    Print("APIレスポンスヘッダ: ", responseHeaders);
    Print("APIレスポンス長さ: ", StringLen(response));
    
    // レスポンスが長い場合は部分的に表示
    if(StringLen(response) > 500) {
        Print("APIレスポンス (最初の500文字): ", StringSubstr(response, 0, 500));
        Print("APIレスポンス (最後の500文字): ", StringSubstr(response, StringLen(response) - 500));
    } else {
        Print("APIレスポンス全体: ", response);
    }
    
    // JSONレスポンスを解析
    ProcessAPIResponse(response);
}

//+------------------------------------------------------------------+
//| 過去60分間のOHLCデータを準備                                     |
//+------------------------------------------------------------------+
string PrepareInputData()
{
    // 60分のOHLCデータを2次元配列として準備 [samples, seq_len, features]
    // サンプルは1つだけ
    string result = "[[[";
    
    for (int i = 60; i >= 1; i--)
    {
        // 過去60本の1分足データを取得
        double open = iOpen(Symbol(), PERIOD_M1, i);
        double high = iHigh(Symbol(), PERIOD_M1, i);
        double low = iLow(Symbol(), PERIOD_M1, i);
        double close = iClose(Symbol(), PERIOD_M1, i);
        
        // データをJSON形式で追加
        if (i < 60) result += "],[";
        result += DoubleToString(open, 3) + "," + 
                  DoubleToString(high, 3) + "," + 
                  DoubleToString(low, 3) + "," + 
                  DoubleToString(close, 3);
    }
    
    result += "]]]";
    return result;
}

//+------------------------------------------------------------------+
//| APIレスポンスを処理して取引シグナルを抽出                          |
//+------------------------------------------------------------------+
void ProcessAPIResponse(string response)
{
    // レスポンスが空ならエラー
    if (StringLen(response) == 0)
    {
        Print("APIレスポンスが空です");
        return;
    }
    
    // まずtrade_suggestionのサブセクションを見つける
    string tradeSuggestionSection = ExtractJsonSection(response, "trade_suggestion");
    if (tradeSuggestionSection == "")
    {
        Print("取引提案セクションが見つかりません");
        return;
    }
    
    Print("trade_suggestion セクション: ", tradeSuggestionSection);
    
    // アクション (BUY/SELL/HOLD) を抽出
    string action = ExtractJsonValue(tradeSuggestionSection, "action");
    if (action == "") 
    {
        Print("取引アクションが見つかりません");
        return;
    }
    
    // 信頼度を抽出
    string confidenceStr = ExtractJsonValue(tradeSuggestionSection, "confidence");
    double confidence = StringToDouble(confidenceStr);
    
    // エントリー価格を抽出
    string entryPriceStr = ExtractJsonValue(tradeSuggestionSection, "entry_price");
    double entryPrice = 0;
    if (entryPriceStr != "null" && entryPriceStr != "")
        entryPrice = StringToDouble(entryPriceStr);
    else
        entryPrice = Ask; // nullの場合は現在の価格を使用
    
    // 目標価格を抽出
    string targetPriceStr = ExtractJsonValue(tradeSuggestionSection, "target_price");
    double targetPrice = 0;
    if (targetPriceStr != "null" && targetPriceStr != "")
        targetPrice = StringToDouble(targetPriceStr);
    else if (action == "BUY")
        targetPrice = Ask * 1.01; // nullの場合は1%上昇を目標に
    else if (action == "SELL")
        targetPrice = Bid * 0.99; // nullの場合は1%下落を目標に
    
    // ストップロス価格を抽出
    string stopLossStr = ExtractJsonValue(tradeSuggestionSection, "stop_loss");
    double stopLoss = 0;
    if (stopLossStr != "null" && stopLossStr != "")
        stopLoss = StringToDouble(stopLossStr);
    else if (action == "BUY")
        stopLoss = Ask * 0.995; // nullの場合は0.5%下落でストップロス
    else if (action == "SELL")
        stopLoss = Bid * 1.005; // nullの場合は0.5%上昇でストップロス
    
    // 分析セクションがあれば取得
    string analysisSection = ExtractJsonSection(tradeSuggestionSection, "analysis");
    
    // 分割エントリーを抽出（デフォルトをtrueに設定）
    bool splitEntry = UseSplitEntry;
    
    Print("取引シグナル: ", action, ", 信頼度: ", confidence, 
          ", エントリー価格: ", entryPrice, ", 目標価格: ", targetPrice, 
          ", ストップロス: ", stopLoss, ", 分割エントリー: ", splitEntry);
    
    // 取引シグナルを保存
    lastTradeSignal = action;
    lastSignalConfidence = confidence;
    
    // 信頼度が閾値を超えていれば取引を実行
    if (confidence >= MinimumConfidence)
    {
        ExecuteTrade(action, entryPrice, targetPrice, stopLoss, splitEntry);
    }
    else
    {
        Print("信頼度が低いため取引を見送ります: ", confidence, " < ", MinimumConfidence);
    }
}

//+------------------------------------------------------------------+
//| JSONからセクションを抽出                                         |
//+------------------------------------------------------------------+
string ExtractJsonSection(string json, string sectionName)
{
    // 正規表現を使わず、まずは単純にキーを探す
    string keyPattern = "\"" + sectionName + "\"";
    int keyPos = StringFind(json, keyPattern);
    
    if (keyPos == -1)
    {
        Print("セクション '" + sectionName + "' が見つかりません");
        return "";
    }
    
    // キーの後のコロンを探す
    int colonPos = StringFind(json, ":", keyPos + StringLen(keyPattern));
    if (colonPos == -1)
    {
        Print("キー '" + sectionName + "' の後にコロンが見つかりません");
        return "";
    }
    
    // コロンの後の空白をスキップ
    int startPos = colonPos + 1;
    while (startPos < StringLen(json) && StringSubstr(json, startPos, 1) == " ")
        startPos++;
    
    // オブジェクト開始の波括弧を探す
    if (startPos >= StringLen(json) || StringSubstr(json, startPos, 1) != "{")
    {
        Print("キー '" + sectionName + "' の値がオブジェクトではありません");
        return "";
    }
    
    // 括弧の対応を追跡
    int braceCount = 1;
    int endPos = startPos;
    
    for (int pos = startPos + 1; pos < StringLen(json) && braceCount > 0; pos++)
    {
        string ch = StringSubstr(json, pos, 1);
        if (ch == "{")
            braceCount++;
        else if (ch == "}")
            braceCount--;
            
        if (braceCount == 0)
            endPos = pos;
    }
    
    if (braceCount > 0)
    {
        Print("JSONの括弧が閉じられていません");
        return "";
    }
    
    // セクション全体を返す（{}を含む）
    return StringSubstr(json, startPos, endPos - startPos + 1);
}

//+------------------------------------------------------------------+
//| 取引を実行                                                       |
//+------------------------------------------------------------------+
void ExecuteTrade(string action, double entryPrice, double targetPrice, double stopLoss, bool splitEntry)
{
    // 既存のポジションをチェック
    int totalPositions = CountOpenPositions();
    
    // HOLDの場合は何もしない
    if (action == "HOLD")
    {
        Print("シグナル: HOLD - 新規ポジションは取りません");
        return;
    }
    
    // 既存のポジションがある場合は決済を検討
    if (totalPositions > 0)
    {
        // 現在の方向と逆のシグナルが出たら決済
        if ((action == "BUY" && HasSellPosition()) || (action == "SELL" && HasBuyPosition()))
        {
            CloseAllPositions();
            Print("方向が変わったため既存ポジションをすべて決済しました");
        }
    }
    
    // ロットサイズを計算
    double lots = CalculateLotSize(action, entryPrice, stopLoss);
    
    // 分割エントリーを使用するかどうかを決定
    int entries = 1;
    if (UseSplitEntry && splitEntry)
    {
        entries = SplitEntryCount;
        lots = NormalizeDouble(lots / entries, 2);
        if (lots < MarketInfo(Symbol(), MODE_MINLOT))
            lots = MarketInfo(Symbol(), MODE_MINLOT);
    }
    
    // 注文を出す
    for (int i = 0; i < entries; i++)
    {
        if (action == "BUY")
        {
            int ticket = OrderSend(Symbol(), OP_BUY, lots, Ask, 10, stopLoss, targetPrice, 
                              "iTransformer", 0, 0, clrGreen);
            if (ticket > 0)
                Print("買い注文が成功しました: チケット=", ticket, ", ロット=", lots);
            else
                Print("買い注文が失敗しました: エラー=", GetLastError());
        }
        else if (action == "SELL")
        {
            int ticket = OrderSend(Symbol(), OP_SELL, lots, Bid, 10, stopLoss, targetPrice, 
                              "iTransformer", 0, 0, clrRed);
            if (ticket > 0)
                Print("売り注文が成功しました: チケット=", ticket, ", ロット=", lots);
            else
                Print("売り注文が失敗しました: エラー=", GetLastError());
        }
        
        // 分割エントリーの場合は少し待つ
        if (entries > 1 && i < entries - 1)
            Sleep(1000); // 1秒待機
    }
}

//+------------------------------------------------------------------+
//| ロットサイズを計算                                               |
//+------------------------------------------------------------------+
double CalculateLotSize(string action, double entryPrice, double stopLoss)
{
    // 固定ロットサイズを使用する場合
    if (UseFixedLotSize)
        return LotSize;
    
    // リスクベースのロットサイズを計算
    double accountBalance = AccountBalance();
    double riskAmount = accountBalance * (RiskPercent / 100.0);
    
    // ストップロスまでのpipを計算
    double pips;
    if (action == "BUY")
        pips = MathAbs(entryPrice - stopLoss) * 100; // USDJPYは2桁（100pips = 1円）
    else
        pips = MathAbs(stopLoss - entryPrice) * 100;
    
    // 1pipあたりの価値を計算
    double pipValue = MarketInfo(Symbol(), MODE_TICKVALUE) * 10; // 0.1pipからpipへの変換
    
    // ロットサイズを計算
    double calculatedLots = riskAmount / (pips * pipValue);
    
    // ロットサイズを正規化
    double normalizedLots = NormalizeDouble(calculatedLots, 2);
    
    // 最小ロットと最大ロットの範囲内に収める
    double minLot = MarketInfo(Symbol(), MODE_MINLOT);
    double maxLot = MarketInfo(Symbol(), MODE_MAXLOT);
    
    if (normalizedLots < minLot) normalizedLots = minLot;
    if (normalizedLots > maxLot) normalizedLots = maxLot;
    
    return normalizedLots;
}

//+------------------------------------------------------------------+
//| 開いているポジションの数を取得                                    |
//+------------------------------------------------------------------+
int CountOpenPositions()
{
    int count = 0;
    for (int i = 0; i < OrdersTotal(); i++)
    {
        if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
        {
            if (OrderSymbol() == Symbol() && (OrderType() == OP_BUY || OrderType() == OP_SELL))
                count++;
        }
    }
    return count;
}

//+------------------------------------------------------------------+
//| 買いポジションがあるかをチェック                                  |
//+------------------------------------------------------------------+
bool HasBuyPosition()
{
    for (int i = 0; i < OrdersTotal(); i++)
    {
        if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
        {
            if (OrderSymbol() == Symbol() && OrderType() == OP_BUY)
                return true;
        }
    }
    return false;
}

//+------------------------------------------------------------------+
//| 売りポジションがあるかをチェック                                  |
//+------------------------------------------------------------------+
bool HasSellPosition()
{
    for (int i = 0; i < OrdersTotal(); i++)
    {
        if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
        {
            if (OrderSymbol() == Symbol() && OrderType() == OP_SELL)
                return true;
        }
    }
    return false;
}

//+------------------------------------------------------------------+
//| すべてのポジションを決済                                         |
//+------------------------------------------------------------------+
void CloseAllPositions()
{
    for (int i = OrdersTotal() - 1; i >= 0; i--)
    {
        if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
        {
            if (OrderSymbol() == Symbol())
            {
                bool result = false;
                if (OrderType() == OP_BUY)
                    result = OrderClose(OrderTicket(), OrderLots(), Bid, 10, clrBlue);
                else if (OrderType() == OP_SELL)
                    result = OrderClose(OrderTicket(), OrderLots(), Ask, 10, clrBlue);
                
                if (!result)
                    Print("ポジション決済に失敗しました: エラー=", GetLastError());
            }
        }
    }
}

//+------------------------------------------------------------------+
//| JSONからキーに対応する値を抽出（改良版）                          |
//+------------------------------------------------------------------+
string ExtractJsonValue(string json, string key)
{
    // 単純なキー文字列を探す
    string keyPattern = "\"" + key + "\"";
    int keyPos = StringFind(json, keyPattern);
    
    if (keyPos == -1)
    {
        Print("キー '" + key + "' が見つかりません");
        return "";
    }
    
    // キーの後のコロンの位置を検索
    int colonPos = StringFind(json, ":", keyPos + StringLen(keyPattern));
    if (colonPos == -1)
    {
        Print("キー '" + key + "' の後にコロンが見つかりません");
        return "";
    }
    
    // コロンの後の空白をスキップ
    int valueStart = colonPos + 1;
    while (valueStart < StringLen(json) && StringSubstr(json, valueStart, 1) == " ")
        valueStart++;
    
    if (valueStart >= StringLen(json))
    {
        Print("キー '" + key + "' の値が見つかりません");
        return "";
    }
    
    string firstChar = StringSubstr(json, valueStart, 1);
    
    // nullの場合
    if (valueStart + 4 <= StringLen(json))
    {
        string possibleNull = StringSubstr(json, valueStart, 4);
        if (possibleNull == "null" || possibleNull == "None")
            return "null";
    }
    
    // 値の終わりの位置を検索
    int valueEnd = valueStart;
    
    // 値が引用符で囲まれているかチェック
    if (firstChar == "\"")
    {
        valueStart++; // 開始引用符をスキップ
        
        // 閉じる引用符を検索（エスケープされていない引用符を探す）
        bool escaped = false;
        for (valueEnd = valueStart; valueEnd < StringLen(json); valueEnd++)
        {
            string ch = StringSubstr(json, valueEnd, 1);
            if (ch == "\\" && !escaped)
            {
                escaped = true;
                continue;
            }
            if (ch == "\"" && !escaped)
                break;
            escaped = false;
        }
        
        if (valueEnd >= StringLen(json))
        {
            Print("閉じる引用符が見つかりません");
            return "";
        }
            
        return StringSubstr(json, valueStart, valueEnd - valueStart);
    }
    else if (firstChar == "{") // オブジェクトの場合
    {
        int braceCount = 1;
        valueEnd = valueStart + 1;
        
        while (valueEnd < StringLen(json) && braceCount > 0)
        {
            string ch = StringSubstr(json, valueEnd, 1);
            if (ch == "{")
                braceCount++;
            else if (ch == "}")
                braceCount--;
                
            valueEnd++;
        }
        
        if (braceCount > 0)
        {
            Print("オブジェクトの閉じ括弧が見つかりません");
            return "";
        }
            
        return StringSubstr(json, valueStart, valueEnd - valueStart);
    }
    else if (firstChar == "[") // 配列の場合
    {
        int bracketCount = 1;
        valueEnd = valueStart + 1;
        
        while (valueEnd < StringLen(json) && bracketCount > 0)
        {
            string ch = StringSubstr(json, valueEnd, 1);
            if (ch == "[")
                bracketCount++;
            else if (ch == "]")
                bracketCount--;
                
            valueEnd++;
        }
        
        if (bracketCount > 0)
        {
            Print("配列の閉じ括弧が見つかりません");
            return "";
        }
            
        return StringSubstr(json, valueStart, valueEnd - valueStart);
    }
    else // 数値や真偽値などの場合
    {
        while (valueEnd < StringLen(json))
        {
            string ch = StringSubstr(json, valueEnd, 1);
            if (ch == "," || ch == "}" || ch == "]")
                break;
            valueEnd++;
        }
    }
    
    return StringSubstr(json, valueStart, valueEnd - valueStart);
}

//+------------------------------------------------------------------+
//| MT4エラーコードから説明文字列を取得                               |
//+------------------------------------------------------------------+
string GetErrorDescription(int errorCode)
{
    switch(errorCode)
    {
        // 一般的なエラー
        case 0: return "エラーなし";
        case 1: return "一般的なエラー";
        case 2: return "共通関数が無効なパラメータで呼び出されました";
        case 3: return "関数のパラメータが無効です";
        case 4: return "メモリ不足";
        case 5: return "構造体が無効です";
        case 6: return "無効な配列";
        case 7: return "配列のサイズが足りません";
        case 8: return "文字列がありません";
        case 9: return "初期化されていない文字列";
        
        // WebRequest関連のエラー
        case 4060: return "WebRequest関数の許可がありません";
        case 4061: return "URL接続に失敗しました";
        case 4062: return "接続に失敗しました";
        case 4063: return "HTTPリクエストのタイムアウト";
        case 4064: return "HTTPレスポンスが無効です";
        
        // 取引関連のエラー
        case 130: return "ソケットが切断されました";
        case 131: return "データ送信エラー";
        case 132: return "データ受信エラー";
        case 133: return "取引内部エラー";
        case 134: return "不十分な資金";
        case 135: return "価格が変更されました";
        case 136: return "価格がありません";
        case 138: return "注文のロック";
        case 139: return "注文が拒否されました";
        case 140: return "注文は既にクローズされています";
        case 146: return "取引サブシステムがビジー状態です";
        
        // デフォルト
        default: return "不明なエラー (" + errorCode + ")";
    }
} 