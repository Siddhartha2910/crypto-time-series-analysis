import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import register_matplotlib_converters
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Ensure date handling in plots
register_matplotlib_converters()

# Set global plot style
plt.style.use('dark_background')

# Set page configuration
st.set_page_config(
    page_title="Matic Cryptocurrency Analysis App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
    <style>
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        
        [data-testid="stSidebar"] {
            background-color: #1A1C24;
            color: white;
            padding: 1rem;
        }

        .main-header {
            font-size: 2.5em;
            font-weight: bold;
            background: linear-gradient(45deg, #FF6347, #FFA500);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
        }

        .sub-header {
            color: #A9A9A9;
            font-size: 1.2em;
            margin-bottom: 2rem;
        }

        .plot-title {
            font-size: 1.5em;
            color: #FFA500;
            font-weight: bold;
            margin: 1rem 0;
        }

        /* Predict button styling with hover effect */
        .stButton>button {
            background-color: #FF4500;
            color: white;
            font-size: 1.2em;
            width: 25%;
            border-radius: 5px;
            padding: 10px 20px;
            box-shadow: 0px 4px 10px rgba(255, 69, 0, 0.3);
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #000000;
        }
        .metric-card {
            background: rgba(26, 28, 36, 0.8);
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .predict-value-container {
            background: linear-gradient(135deg, #1E1E1E, #2A2A2A);
            border-radius: 12px;
            padding: 2rem;
            margin: 1.5rem 0;
            box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(0, 204, 102, 0.2);
        }

        .predict-value {
            font-size: 2.5em;
            color: #00cc66;
            text-align: center;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

class MaticAnalyzer:
    def __init__(self):
        self.data = None
        
    def load_data(self, filepath):
        """Load and preprocess the Matic data"""
        try:
            data = pd.read_csv(filepath)
            data['Date'] = pd.to_datetime(data['Date'], format='%b %d, %Y')
            data['Vol.'] = data['Vol.'].replace({'M': 'e6', 'B': 'e9'}, regex=True)
            data['Vol.'] = pd.to_numeric(data['Vol.'], errors='coerce')
            data['Change %'] = data['Change %'].str.replace('%', '').astype(float)
            self.data = data.sort_values('Date')
            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False

    def calculate_metrics(self):
        """Calculate key trading metrics"""
        metrics = {
            'Current Price': self.data['Price'].iloc[-1],
            'Daily Return': self.data['Change %'].mean(),
            'Volatility': self.data['Change %'].std(),
            'Highest Price': self.data['Price'].max(),
            'Lowest Price': self.data['Price'].min(),
            'Average Volume': self.data['Vol.'].mean()
        }
        return metrics

    def calculate_volatility(self, price_data, window=30):
        """Calculate price volatility using returns"""
        returns = np.diff(price_data) / price_data[:-1]
        return np.std(returns[-window:]) if len(returns) >= window else np.std(returns)

    def predict_price(self, future_date):
        """Enhanced ARIMA prediction with volatility adjustment"""
        days_to_forecast = (future_date - self.data['Date'].max()).days
        if days_to_forecast <= 0:
            return None, None, None, None

        try:
            price_data = self.data['Price'].values
            
            # Calculate volatility using returns
            volatility = self.calculate_volatility(price_data)

            # Dynamic ARIMA parameters
            p = min(5, days_to_forecast // 5)
            d = 1
            q = min(2, days_to_forecast // 10)

            model = ARIMA(price_data, order=(p, d, q))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=days_to_forecast)
            
            # Add controlled randomness
            random_factors = np.random.normal(1, volatility/2, len(forecast))
            forecast = forecast * random_factors
            
            # Calculate confidence intervals
            std_error = volatility * np.sqrt(days_to_forecast)
            forecast_value = forecast[-1]
            lower_bound = forecast_value * (1 - 2 * std_error)
            upper_bound = forecast_value * (1 + 2 * std_error)
            
            return forecast_value, lower_bound, upper_bound, volatility

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None, None, None, None

# Function to plot price over time
def plot_price_over_time(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Date'], data['Price'], label='Matic Price', color='#ff6b6b')
    ax.set_title('Matic Cryptocurrency Price Over Time', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig

# Function to plot moving averages (30-Day and 90-Day)
def plot_moving_averages(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Date'], data['Price'], label='Price', color='#ff6b6b')
    ax.plot(data['Date'], data['MA30'], label='30-Day Moving Average', color='#4da6ff')
    ax.plot(data['Date'], data['MA90'], label='90-Day Moving Average', color='#00cc66')
    ax.set_title('Matic Price with 30-Day and 90-Day Moving Averages', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig

# Function to plot trading volume
def plot_volume(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(data['Date'], data['Vol.'], color='#4da6ff', alpha=0.7)
    ax.set_title('Matic Trading Volume Over Time', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Volume')
    ax.grid(True, alpha=0.3)
    return fig
# Visualization Functions
def plot_price_distribution(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(data['Price'], kde=True, color='#FF6347', bins=30)
    ax.set_title('Price Distribution of Matic', fontsize=14)
    ax.set_xlabel('Price (USD)')
    ax.set_ylabel('Frequency')
    return fig

def plot_correlation_heatmap(data):
    corr = data[['Price', 'Vol.']].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
    ax.set_title('Correlation Heatmap between Price and Volume', fontsize=14)
    return fig

def plot_rolling_statistics(data, window=30):
    fig, ax = plt.subplots(figsize=(12, 6))
    rolling_mean = data['Price'].rolling(window=window).mean()
    rolling_std = data['Price'].rolling(window=window).std()
    
    ax.plot(data['Date'], rolling_mean, label=f'{window}-Day Rolling Mean', color='#4da6ff')
    ax.plot(data['Date'], rolling_std, label=f'{window}-Day Rolling Std Dev', color='#FF6347')
    
    ax.set_title(f'Rolling Statistics ({window}-Day) of Matic Price', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig

def plot_boxplot(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x=data['Price'], color='#4da6ff', ax=ax)
    ax.set_title('Boxplot of Matic Price Distribution', fontsize=14)
    ax.set_xlabel('Price (USD)')
    return fig

def plot_price_over_time(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Date'], data['Price'], color='#FF6347', label='Price (USD)')
    ax.set_title('Matic Cryptocurrency Price Over Time', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig

def plot_volume_over_time(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Date'], data['Vol.'], color='#1f77b4', label='Volume')
    ax.set_title('Matic Trading Volume Over Time', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Volume')
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def main():
    analyzer = MaticAnalyzer()
    
    st.markdown('<h1 class="main-header">🚀 Matic Cryptocurrency Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced technical analysis and price prediction</p>', unsafe_allow_html=True)

    # Load data
    if not analyzer.load_data('C:/Users/gsidd/Downloads/Matic Historical Data.csv'):
        st.stop()

    # Create tabs
    tabs = st.tabs(["🔮 Prediction", "📊 Analysis", "📈 Charts"])

    with tabs[0]:
        st.markdown("### Price Prediction")
        future_date = st.date_input(
            "Select forecast date",
            min_value=analyzer.data['Date'].max() + pd.Timedelta(days=1)
        )
        
        if st.button("Generate Forecast", key="predict_button"):
            prediction, lower, upper, volatility = analyzer.predict_price(pd.to_datetime(future_date))
            
            if prediction is not None:
                st.markdown(
                    f"""
                    <div class="predict-value-container">
                        <h3 style="text-align: center; color: #00cc66;">Predicted Price</h3>
                        <div class="predict-value">${prediction:.4f}</div>
                        <p style="text-align: center; color: white;">Volatility: {volatility:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                st.balloons()
                
                st.markdown(f"### 95% Confidence Interval: ${lower:.4f} - ${upper:.4f}")

    with tabs[1]:
        st.subheader("📈 Matic Cryptocurrency Metrics")
        metrics = analyzer.calculate_metrics()
        
        for key, value in metrics.items():
            st.metric(label=key, value=f"${value:,.2f}" if isinstance(value, (int, float)) else f"{value}%")
        
        st.markdown("### Price Distribution")
        fig = plot_price_distribution(analyzer.data)
        st.pyplot(fig)

        st.markdown("### Correlation Heatmap")
        fig = plot_correlation_heatmap(analyzer.data)
        st.pyplot(fig)

        st.markdown("### Rolling Statistics (30-Day Moving Average and Std Dev)")
        fig = plot_rolling_statistics(analyzer.data, window=30)
        st.pyplot(fig)

        st.markdown("### Boxplot of Price Distribution")
        fig = plot_boxplot(analyzer.data)
        st.pyplot(fig)

        # Insights on analysis
        st.markdown(
            """
            #### Insights:
            - The price distribution shows the range of typical prices for Matic.
            - The correlation heatmap gives insights into how price and volume are related.
            - The rolling statistics can help us see how the price has fluctuated over the past 30 days.
            - The boxplot shows any outliers in the price data, indicating unusual price movements.
            """
        )

    with tabs[2]:
        st.subheader("📈 Matic Cryptocurrency Price Over Time")
        fig = plot_price_over_time(analyzer.data)
        st.pyplot(fig)

        # Moving Averages
        analyzer.data['MA30'] = analyzer.data['Price'].rolling(window=30).mean()
        analyzer.data['MA90'] = analyzer.data['Price'].rolling(window=90).mean()

        st.subheader("📊 Price with 30-Day and 90-Day Moving Averages")
        fig = plot_moving_averages(analyzer.data)
        st.pyplot(fig)

        # Volume Plot
        st.subheader("📉 Trading Volume Over Time")
        fig = plot_volume(analyzer.data)
        st.pyplot(fig)

        # Drop MA30 and MA90 columns before displaying the dataset preview
        dataset_preview = analyzer.data.drop(columns=['MA30', 'MA90'])
        st.subheader("📊 Dataset Preview")
        st.dataframe(dataset_preview.tail())  # Displaying last few rows as a preview

if __name__ == "__main__":
    main()
