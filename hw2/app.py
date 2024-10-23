# 必要的庫導入
import os
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from flask import Flask, render_template

# 建立 Flask 應用
app = Flask(__name__)

# 預測及繪圖函數
def plot_forecast(output_path):
    # 讀取資料
    file_path = 'data/data.csv'
    df = pd.read_csv(file_path)

    # 資料處理
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df['y'] = df['y'].replace({',': ''}, regex=True).astype(float)
    df = df[['Date', 'y']].rename(columns={'Date': 'ds'})

    # Prophet 模型
    model = Prophet(changepoint_prior_scale=0.5, interval_width=0.95)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.fit(df)

    # 預測未來 60 天
    future = model.make_future_dataframe(periods=60)
    forecast = model.predict(future)

    # 畫圖
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['ds'], df['y'], 'k-', label='Actual Stock Price')
    ax.plot(forecast['ds'], forecast['yhat'], 'b-', label='Forecasted Stock Price')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='blue', alpha=0.2)
    
    # 添加歷史平均線
    historical_avg = df['y'].mean()
    ax.axhline(y=historical_avg, color='gray', linestyle='--', label='Historical Average')
    
    # 添加箭頭標註
    last_date = df['ds'].iloc[-1]
    last_price = df['y'].iloc[-1]
    ax.annotate('Last Actual Point', xy=(last_date, last_price), xytext=(last_date, last_price + 50),
                arrowprops=dict(facecolor='red', shrink=0.05), fontsize=12, color='red')

    plt.title('Stock Price Forecast with Prophet')
    plt.legend()
    
    # 保存圖表到指定路徑
    plt.savefig(output_path)
    plt.close()

# 路由處理
@app.route('/')
def index():
    # 產生預測圖表
    plot_path = os.path.join('static', 'forecast.png')
    plot_forecast(plot_path)
    
    # 渲染首頁，並將圖表路徑傳遞給模板
    return render_template('index.html', plot_url=plot_path)

# 啟動 Flask 應用
if __name__ == "__main__":
    app.run(debug=True)
