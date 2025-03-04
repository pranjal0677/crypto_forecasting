import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import requests
from models.prophet_model import ProphetModel
from models.arima_model import ArimaModel
from models.lstm_model import LSTMModel
from utils.preprocessing import preprocess_data
from utils.visualization import plot_decomposition, plot_forecast

st.set_page_config(page_title="Crypto Forecasting Platform", layout="wide")

class CryptoForecastingApp:
    def __init__(self):
        self.available_coins = {
            'Bitcoin': 'BTCUSDT',
            'Ethereum': 'ETHUSDT',
            'Dogecoin': 'DOGEUSDT'
        }
        #self.available_models = ['Prophet', 'ARIMA', 'LSTM']
        self.available_models = ['Prophet', 'ARIMA'] 

    def get_binance_data(self, symbol, start_date, end_date):
        try:
            # Convert datetime.date to datetime.datetime
            start_datetime = datetime.combine(start_date, datetime.min.time())
            end_datetime = datetime.combine(end_date, datetime.max.time())
            
            # Convert to millisecond timestamps
            start_ts = int(start_datetime.timestamp() * 1000)
            end_ts = int(end_datetime.timestamp() * 1000)
            
            # Binance API endpoint
            url = "https://api.binance.com/api/v3/klines"
            
            # Parameters for the API request
            params = {
                "symbol": symbol,
                "interval": "1d",
                "startTime": start_ts,
                "endTime": end_ts,
                "limit": 1000
            }
            
            response = requests.get(url, params=params)
            
            # Check if the response is successful
            if response.status_code != 200:
                st.error(f"Error from Binance API: {response.text}")
                return None
                
            data = response.json()
            
            # Check if we got any data
            if not data:
                st.warning("No data returned from Binance for the selected period")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                           'close_time', 'quote_asset_volume', 'number_of_trades',
                                           'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                                           'ignore'])
            
            # Convert timestamp to datetime
            df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('Date', inplace=True)
            
            # Convert string values to float
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Rename columns to match our expected format
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            # Sort index to ensure chronological order
            df = df.sort_index()
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            st.error(f"Error fetching data from Binance: {str(e)}")
            return None

    def run(self):
        st.title("Cryptocurrency Price Forecasting & Analysis Platform")

        # Sidebar controls
        st.sidebar.header("Controls")
        selected_coin_name = st.sidebar.selectbox('Choose Cryptocurrency', list(self.available_coins.keys()))
        selected_coin = self.available_coins[selected_coin_name]
        
        # Date range selection
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                datetime.now().date() - timedelta(days=365),
                min_value=datetime(2017, 1, 1).date(),
                max_value=datetime.now().date()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                datetime.now().date(),
                min_value=start_date,
                max_value=datetime.now().date()
            )

        if start_date >= end_date:
            st.error("Error: End date must be after start date.")
            return

        selected_model = st.sidebar.selectbox('Choose Model', self.available_models)
        forecast_days = st.sidebar.slider('Forecast Days', 7, 90, 30)

        if st.sidebar.button('Generate Forecast'):
            with st.spinner('Loading data from Binance...'):
                data = self.get_binance_data(selected_coin, start_date, end_date)

            if data is not None and not data.empty:
                try:
                    with st.spinner('Processing data...'):
                        processed_data = preprocess_data(data)

                    # Display basic stats
                    st.subheader("Current Statistics")
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    
                    with metrics_col1:
                        current_price = data['Close'][-1]
                        st.metric("Current Price", f"${current_price:.2f}")
                    
                    with metrics_col2:
                        if len(data) > 1:
                            price_change = data['Close'][-1] - data['Close'][-2]
                            price_change_pct = (price_change / data['Close'][-2]) * 100
                            st.metric("24h Change", 
                                    f"${price_change:.2f}",
                                    f"{price_change_pct:.2f}%")
                        else:
                            st.metric("24h Change", "N/A")
                    
                    with metrics_col3:
                        current_volume = data['Volume'][-1]
                        st.metric("24h Volume", f"${current_volume:,.0f}")

                    # Show decomposition
                    st.subheader("Time Series Decomposition")
                    with st.spinner('Generating decomposition plot...'):
                        try:
                            decomp_fig = plot_decomposition(processed_data)
                            st.plotly_chart(decomp_fig)
                        except Exception as e:
                            st.error(f"Error in decomposition: {str(e)}")

                    # Generate and show forecast
                    st.subheader(f"Price Forecast ({forecast_days} days)")
                    with st.spinner(f'Training {selected_model} model and generating forecast...'):
                        try:
                            if selected_model == 'Prophet':
                                model = ProphetModel()
                            elif selected_model == 'ARIMA':
                                model = ArimaModel()
                            else:
                                model = LSTMModel()

                            forecast = model.train_and_predict(processed_data, forecast_days)
                            forecast_fig = plot_forecast(processed_data, forecast, selected_model)
                            st.plotly_chart(forecast_fig)

                            # Show model performance metrics
                            st.subheader("Model Performance Metrics")
                            metrics = model.get_metrics()
                            metrics_df = pd.DataFrame(metrics, index=[0])
                            st.table(metrics_df)
                        
                        except Exception as e:
                            st.error(f"Error in forecasting: {str(e)}")

                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")
            else:
                st.warning("No data available for the selected date range. Please try a different range.")

        # Add information about data source
        st.sidebar.markdown("---")
        st.sidebar.info("Data source: Binance API")

        with st.expander("How to use this app"):
            st.write("""
            1. Select a cryptocurrency from the dropdown menu
            2. Choose your desired date range (data available from 2017 onwards)
            3. Select a forecasting model
            4. Choose the number of days to forecast
            5. Click 'Generate Forecast' to see the results
            
            Note: Data is sourced from Binance exchange and is limited to 1000 data points per request.
            """)

if __name__ == "__main__":
    app = CryptoForecastingApp()
    app.run()