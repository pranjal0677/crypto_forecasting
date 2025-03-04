import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose

def plot_decomposition(data):
    decomposition = seasonal_decompose(data['Close'], period=30)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=decomposition.trend,
        name="Trend"
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=decomposition.seasonal,
        name="Seasonal"
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=decomposition.resid,
        name="Residual"
    ))
    
    fig.update_layout(
        title="Time Series Decomposition",
        xaxis_title="Date",
        yaxis_title="Price",
        height=800
    )
    
    return fig

def plot_forecast(historical_data, forecast, model_name):
    fig = go.Figure()
    
    # Plot historical data
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data['Close'],
        name="Historical"
    ))
    
    # Plot forecast
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        name="Forecast",
        line=dict(dash='dash')
    ))
    
    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        fill=None,
        mode='lines',
        line_color='rgba(0,100,80,0.2)',
        name='Upper Bound'
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,100,80,0.2)',
        name='Lower Bound'
    ))
    
    fig.update_layout(
        title=f"{model_name} Forecast",
        xaxis_title="Date",
        yaxis_title="Price",
        height=600
    )
    
    return fig