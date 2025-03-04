import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error

class LSTMModel:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.metrics = {}
        
    def create_sequences(self, data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)
    
    def train_and_predict(self, data, forecast_days):
        # Prepare data
        scaled_data = self.scaler.fit_transform(data[['Close']].values)
        
        # Create sequences
        seq_length = 60
        X, y = self.create_sequences(scaled_data, seq_length)
        
        # Create and train model
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(X, y, epochs=20, batch_size=32, verbose=0)
        
        # Make predictions
        last_sequence = scaled_data[-seq_length:]
        future_predictions = []
        
        for _ in range(forecast_days):
            next_pred = self.model.predict(last_sequence.reshape(1, seq_length, 1), verbose=0)
            future_predictions.append(next_pred[0, 0])
            last_sequence = np.roll(last_sequence, -1)
            last_sequence[-1] = next_pred
            
        # Inverse transform predictions
        future_predictions = np.array(future_predictions).reshape(-1, 1)
        future_predictions = self.scaler.inverse_transform(future_predictions)
        
        # Calculate metrics
        y_true = data['Close'].values[seq_length:]
        y_pred = self.scaler.inverse_transform(self.model.predict(X, verbose=0))
        
        self.metrics = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAPE': np.mean(np.abs((y_true - y_pred.flatten()) / y_true)) * 100
        }
        
        # Create forecast DataFrame
        last_date = data.index[-1]
        forecast_dates = pd.date_range(start=last_date, periods=forecast_days + 1)[1:]
        forecast_df = pd.DataFrame({
            'ds': forecast_dates,
            'yhat': future_predictions.flatten(),
            'yhat_lower': future_predictions.flatten() * 0.9,  # Simple confidence interval
            'yhat_upper': future_predictions.flatten() * 1.1   # Simple confidence interval
        })
        
        return forecast_df

    def get_metrics(self):
        return self.metrics