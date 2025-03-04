import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

class ProphetModel:
    def __init__(self):
        self.model = None
        self.metrics = {}
        self.scaler = None

    def prepare_data(self, data):
        return data['Close'].values

    def train_and_predict(self, data, forecast_days):
        # Prepare data
        train_data = self.prepare_data(data)
        
        # Train model using Holt-Winters' method
        self.model = ExponentialSmoothing(
            train_data,
            seasonal_periods=7,  # Weekly seasonality
            trend='add',
            seasonal='add',
        ).fit()
        
        # Make prediction
        forecast = self.model.forecast(forecast_days)
        
        # Calculate metrics
        y_true = train_data
        y_pred = self.model.fittedvalues
        
        self.metrics = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        
        # Create forecast DataFrame
        last_date = data.index[-1]
        forecast_dates = pd.date_range(start=last_date, periods=forecast_days + 1)[1:]
        
        # Calculate confidence intervals
        confidence_interval = 1.96 * np.std(y_true - y_pred)
        
        forecast_df = pd.DataFrame({
            'ds': forecast_dates,
            'yhat': forecast,
            'yhat_lower': forecast - confidence_interval,
            'yhat_upper': forecast + confidence_interval
        })
        
        return forecast_df

    def get_metrics(self):
        return self.metrics