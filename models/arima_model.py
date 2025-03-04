from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np

class ArimaModel:
    def __init__(self):
        self.model = None
        self.metrics = {}

    def train_and_predict(self, data, forecast_days):
        try:
            # Prepare data
            train_data = data['Close'].values
            
            # Train model (using simple ARIMA parameters, can be optimized)
            self.model = SARIMAX(train_data, 
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12))
            
            fitted_model = self.model.fit(disp=False)
            
            # Make prediction
            forecast = fitted_model.forecast(steps=forecast_days)
            
            # Calculate metrics
            y_true = train_data
            y_pred = fitted_model.get_prediction(start=0).predicted_mean
            
            self.metrics = {
                'MAE': mean_absolute_error(y_true, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            }
            
            # Calculate forecast standard error for confidence intervals
            forecast_std = np.sqrt(fitted_model.get_forecast(steps=forecast_days).var_pred_mean)
            
            # Create forecast DataFrame
            last_date = data.index[-1]
            forecast_dates = pd.date_range(start=last_date, periods=forecast_days + 1)[1:]
            
            # Use 95% confidence intervals (1.96 standard deviations)
            forecast_df = pd.DataFrame({
                'ds': forecast_dates,
                'yhat': forecast,
                'yhat_lower': forecast - (1.96 * forecast_std),
                'yhat_upper': forecast + (1.96 * forecast_std)
            })
            
            return forecast_df
            
        except Exception as e:
            print(f"Error in ARIMA model: {str(e)}")
            # Fallback to simpler forecasting if ARIMA fails
            return self.simple_forecast(data, forecast_days)

    def simple_forecast(self, data, forecast_days):
        """Fallback method using simple exponential smoothing"""
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        # Prepare data
        train_data = data['Close'].values
        
        # Train model
        model = ExponentialSmoothing(
            train_data,
            trend='add'
        ).fit()
        
        # Make prediction
        forecast = model.forecast(forecast_days)
        
        # Calculate metrics
        y_true = train_data
        y_pred = model.fittedvalues
        
        self.metrics = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        
        # Calculate confidence intervals
        std_residuals = np.std(y_true - y_pred)
        
        # Create forecast DataFrame
        last_date = data.index[-1]
        forecast_dates = pd.date_range(start=last_date, periods=forecast_days + 1)[1:]
        
        forecast_df = pd.DataFrame({
            'ds': forecast_dates,
            'yhat': forecast,
            'yhat_lower': forecast - (1.96 * std_residuals),
            'yhat_upper': forecast + (1.96 * std_residuals)
        })
        
        return forecast_df

    def get_metrics(self):
        return self.metrics