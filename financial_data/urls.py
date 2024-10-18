from django.urls import path
from . import views

urlpatterns = [
    path('backtest/', views.backtest, name='backtest'),
    path('predict/', views.predict_stock, name='predict_stock'),  # Prediction URL
    path('generate_report/<str:symbol>/', views.generate_pdf_report, name='generate_pdf_report'),

]
