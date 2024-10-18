from django.db import models

# Create your models here.

class StockData(models.Model):
    symbol = models.CharField(max_length=10)
    date = models.DateField()
    open_price = models.FloatField()
    close_price = models.FloatField()
    high_price = models.FloatField()
    low_price = models.FloatField()
    volume = models.BigIntegerField()

    def __str__(self):
        return f"{self.symbol} - {self.date}"

class StockPrediction(models.Model):
    symbol = models.CharField(max_length=10)
    date = models.DateField()
    predicted_price = models.FloatField()

    def __str__(self):
        return f"{self.symbol} - {self.date}: {self.predicted_price}"