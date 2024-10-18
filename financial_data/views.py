from django.http import JsonResponse, HttpResponse
from .models import StockData, StockPrediction
import pandas as pd
import pickle
from datetime import timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import requests

def backtest(request):
    # Get parameters from the request (default values adjusted)
    initial_investment = float(request.GET.get('investment', 10000))
    buy_threshold = int(request.GET.get('buy_threshold', 20))  # Shorter buy threshold
    sell_threshold = int(request.GET.get('sell_threshold', 50))  # Shorter sell threshold

    # Fetch stock data from the database (same as before)
    stock_data = StockData.objects.filter(symbol='AAPL').order_by('date')

    # Convert data into a DataFrame
    df = pd.DataFrame(list(stock_data.values('date', 'close_price')))
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Calculate shorter moving averages
    df['20_MA'] = df['close_price'].rolling(window=buy_threshold).mean()
    df['50_MA'] = df['close_price'].rolling(window=sell_threshold).mean()

    # Initialize backtest variables (same as before)
    cash = initial_investment
    stock_held = 0
    trades = 0
    total_value = initial_investment

    # Apply the modified buy/sell strategy
    for i in range(len(df)):
        if pd.notna(df['20_MA'].iloc[i]) and pd.notna(df['50_MA'].iloc[i]):
            # Buy when the price is slightly below the 20-day MA
            if df['close_price'].iloc[i] < df['20_MA'].iloc[i] * 0.99 and stock_held == 0:
                stock_held = cash / df['close_price'].iloc[i]
                cash = 0
                trades += 1
            # Sell when the price is slightly above the 50-day MA
            elif df['close_price'].iloc[i] > df['50_MA'].iloc[i] * 1.01 and stock_held > 0:
                cash = stock_held * df['close_price'].iloc[i]
                stock_held = 0
                trades += 1

    # Final portfolio value and ROI calculation (same as before)
    if stock_held > 0:
        total_value = stock_held * df['close_price'].iloc[-1]
    else:
        total_value = cash

    roi = ((total_value - initial_investment) / initial_investment) * 100

    # Return the backtest results
    return JsonResponse({
        "initial_investment": initial_investment,
        "final_portfolio_value": total_value,
        "roi": roi,
        "trades": trades,
    })


def predict_stock(request):
    symbol = request.GET.get('symbol', 'AAPL')

    # Load the historical stock data
    stock_data = StockData.objects.filter(symbol=symbol).order_by('date')
    df = pd.DataFrame(list(stock_data.values('date', 'close_price')))
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Load the pre-trained model
    with open('linear_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Prepare data for prediction (next 30 days)
    last_date = df.index.max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
    future_days = [(date - df.index.min()).days for date in future_dates]

    # Predict future prices
    predictions = model.predict(pd.DataFrame(future_days, columns=['days']))

    # Store predictions in a dictionary for output
    predicted_data = {
        'symbol': symbol,
        'predictions': []
    }

    # Save predictions in the StockPrediction model and add them to the JSON response
    for date, price in zip(future_dates, predictions):
        # Save each prediction in the database
        StockPrediction.objects.update_or_create(
            symbol=symbol,
            date=date,
            defaults={'predicted_price': price}
        )

        predicted_data['predictions'].append({
            'date': date.strftime('%Y-%m-%d'),
            'predicted_price': price
        })

    # Return the predictions as JSON
    return JsonResponse(predicted_data)

def generate_chart(symbol):
    # Fetch actual and predicted stock data
    stock_data = StockData.objects.filter(symbol=symbol).order_by('date')
    df_actual = pd.DataFrame(list(stock_data.values('date', 'close_price')))
    df_actual['date'] = pd.to_datetime(df_actual['date'])
    df_actual.set_index('date', inplace=True)

    stock_predictions = StockPrediction.objects.filter(symbol=symbol).order_by('date')
    df_predicted = pd.DataFrame(list(stock_predictions.values('date', 'predicted_price')))
    df_predicted['date'] = pd.to_datetime(df_predicted['date'])
    df_predicted.set_index('date', inplace=True)

    # Generate plot
    plt.figure(figsize=(10, 6))
    plt.plot(df_actual.index, df_actual['close_price'], label='Actual Price')
    plt.plot(df_predicted.index, df_predicted['predicted_price'], label='Predicted Price', linestyle='--')

    plt.title(f'{symbol} - Actual vs Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    # Save the plot to a BytesIO object
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    return buffer  # Return buffer without using PIL here


def generate_pdf_report(request, symbol):
    # Create a PDF response
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{symbol}_report.pdf"'

    # Set up the PDF canvas
    p = canvas.Canvas(response, pagesize=letter)
    width, height = letter

    # Write the title
    p.setFont("Helvetica-Bold", 16)
    p.drawString(100, height - 50, f'Report for {symbol}')

    # Fetch dynamic backtest results by calling the existing backtest API
    backtest_url = f'http://127.0.0.1:8000/financial_data/backtest/?symbol={symbol}'  # Adjust this URL to match your backtest API
    backtest_response = requests.get(backtest_url)

    if backtest_response.status_code == 200:
        backtest_results = backtest_response.json()

        # Extract metrics from the backtest results
        total_return = backtest_results.get('roi', 0)  # ROI (return on investment)
        trades = backtest_results.get('trades', 0)
    else:
        # If the backtest API fails, use default values
        total_return = 0
        trades = 0

    # Add dynamic key metrics to the PDF
    p.setFont("Helvetica", 12)
    p.drawString(100, height - 100, "Key Performance Metrics:")
    p.drawString(100, height - 120, f"Total Return: {total_return:.2f}%")
    p.drawString(100, height - 140, f"Number of Trades: {trades}")

    # Generate and embed the chart (as we did earlier)
    chart_buffer = generate_chart(symbol)
    image_reader = ImageReader(chart_buffer)
    p.drawImage(image_reader, 100, height - 400, width=400, height=200)

    # Save the PDF
    p.showPage()
    p.save()

    return response