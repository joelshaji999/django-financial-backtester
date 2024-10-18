import os
import requests
from django.core.management.base import BaseCommand
from financial_data.models import StockData
from dotenv import load_dotenv
from datetime import datetime

# Load API key from .env file
load_dotenv()

class Command(BaseCommand):
    help = 'Fetch stock data from Alpha Vantage API'

    def handle(self, *args, **kwargs):
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        symbol = 'AAPL'  # Example stock symbol
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}'

        response = requests.get(url)
        if response.status_code != 200:
            self.stdout.write(self.style.ERROR('Failed to fetch data'))
            return

        data = response.json().get("Time Series (Daily)", {})
        if not data:
            self.stdout.write(self.style.ERROR('No data found'))
            return

        # Save data to the database
        for date, info in data.items():
            StockData.objects.update_or_create(
                symbol=symbol,
                date=datetime.strptime(date, '%Y-%m-%d').date(),
                defaults={
                    'open_price': info["1. open"],
                    'close_price': info["4. close"],
                    'high_price': info["2. high"],
                    'low_price': info["3. low"],
                    'volume': info["5. volume"],
                }
            )

        self.stdout.write(self.style.SUCCESS(f'Successfully fetched and stored data for {symbol}'))
