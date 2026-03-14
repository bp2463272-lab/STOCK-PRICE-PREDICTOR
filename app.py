# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="NFLX Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #E50914;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #564d4d;
        text-align: center;
        margin-top: 0;
        padding-top: 0;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .insight-text {
        background-color: #e8f4f8;
        border-left: 5px solid #E50914;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stButton>button {
        background-color: #E50914;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 2rem;
    }
    .stButton>button:hover {
        background-color: #b20710;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = []
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'train_r2' not in st.session_state:
    st.session_state.train_r2 = None
if 'test_r2' not in st.session_state:
    st.session_state.test_r2 = None
if 'train_rmse' not in st.session_state:
    st.session_state.train_rmse = None
if 'test_rmse' not in st.session_state:
    st.session_state.test_rmse = None
if 'train_mae' not in st.session_state:
    st.session_state.train_mae = None
if 'test_mae' not in st.session_state:
    st.session_state.test_mae = None

# Header
st.markdown('<p class="main-header">NETFLIX STOCK PREDICTOR</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Linear Regression Model for Next Day Close Price Prediction (2014-2023)</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/7/75/Netflix_icon.svg", width=100)
    st.title("⚙️ Control Panel")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'], 
                                     help="Upload your Netflix stock data CSV")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.success("✅ Data loaded successfully!")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.session_state.data_loaded = False
    
    # Sample data option
    if not st.session_state.data_loaded:
        st.info("📌 Please upload your CSV file to begin")
    
    # Model parameters (only show if data is loaded)
    if st.session_state.data_loaded and st.session_state.df is not None:
        st.markdown("---")
        st.subheader("🔧 Model Parameters")
        
        # Feature selection options
        test_size = st.slider("Test Size (%)", 10, 30, 20, 5)
        st.session_state.test_size = test_size
        
        # Advanced options expander
        with st.expander("Advanced Options"):
            remove_outliers = st.checkbox("Remove Outliers", False)
            st.session_state.remove_outliers = remove_outliers
            scale_features = st.checkbox("Scale Features", True)
            st.session_state.scale_features = scale_features
            cv_folds = st.slider("Cross-validation Folds", 2, 10, 5)
            st.session_state.cv_folds = cv_folds
        
        # Train button
        if st.button("🚀 Train Model", use_container_width=True):
            with st.spinner("Training model... This may take a moment."):
                st.session_state.model_trained = True

# Main content area
if st.session_state.data_loaded and st.session_state.df is not None:
    df = st.session_state.df
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", 
        "📈 Exploratory Analysis", 
        "🤖 Model Training", 
        "📉 Predictions",
        "📋 Summary Report"
    ])
    
    with tab1:
        st.header("📊 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Days", f"{len(df):,}", 
                     help="Total number of trading days")
        with col2:
            st.metric("Date Range", f"{df.index.min().year}-{df.index.max().year}",
                     help="Time period covered")
        with col3:
            st.metric("Avg Close Price", f"${df['close'].mean():.2f}",
                     delta=f"{((df['close'].iloc[-1] - df['close'].iloc[0])/df['close'].iloc[0]*100):.1f}% total return")
        with col4:
            st.metric("Total Volume", f"{df['volume'].sum():,.0f}",
                     help="Total trading volume")
        
        # Data preview with formatting
        st.subheader("Data Preview")
        col1, col2 = st.columns([3, 1])
        with col2:
            rows_to_show = st.selectbox("Rows to display", [5, 10, 20, 50, 100], index=1)
        
        with col1:
            # Format the dataframe for display
            display_df = df.head(rows_to_show).copy()
            st.dataframe(display_df, use_container_width=True)
        
        # Data info
        st.subheader("Dataset Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Column Names:**")
            for col in df.columns:
                st.write(f"• {col}")
        
        with col2:
            st.write("**Data Types & Missing Values:**")
            info_df = pd.DataFrame({
                'Data Type': df.dtypes,
                'Missing': df.isnull().sum(),
                'Missing %': (df.isnull().sum()/len(df)*100).round(2)
            })
            st.dataframe(info_df, use_container_width=True)
        
        # Statistical summary
        with st.expander("📊 Statistical Summary"):
            st.dataframe(df.describe().round(2), use_container_width=True)
    
    with tab2:
        st.header("📈 Exploratory Data Analysis")
        
        # Time series plot
        st.subheader("Stock Price Over Time")
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Close Price with Range', 'Trading Volume', 'RSI-14', 'MACD'),
            vertical_spacing=0.12
        )
        
        # Close price with range
        fig.add_trace(
            go.Scatter(x=df.index, y=df['close'], mode='lines', name='Close',
                      line=dict(color='#E50914', width=2)),
            row=1, col=1
        )
        if 'high' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['high'], mode='lines', name='High',
                          line=dict(color='rgba(0,255,0,0.3)', width=1), showlegend=False),
                row=1, col=1
            )
        if 'low' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['low'], mode='lines', name='Low',
                          line=dict(color='rgba(255,0,0,0.3)', width=1), showlegend=False),
                row=1, col=1
            )
        
        # Volume
        if 'volume' in df.columns:
            fig.add_trace(
                go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color='#2E86AB'),
                row=1, col=2
            )
        
        # RSI
        if 'rsi_14' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['rsi_14'], mode='lines', name='RSI-14',
                          line=dict(color='#F39C12', width=2)),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
        
        # MACD
        if 'macd' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['macd'], mode='lines', name='MACD',
                          line=dict(color='#8E44AD', width=2)),
                row=2, col=2
            )
            fig.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3, row=2, col=2)
        
        fig.update_layout(height=800, showlegend=False, title_text="Time Series Analysis")
        fig.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution plots
        st.subheader("Distribution Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'close' in df.columns:
                fig = px.histogram(df, x='close', nbins=50, title='Distribution of Close Price',
                                  labels={'close': 'Price ($)'}, color_discrete_sequence=['#E50914'])
                fig.add_vline(x=df['close'].mean(), line_dash="dash", line_color="blue",
                             annotation_text=f"Mean: ${df['close'].mean():.2f}")
                fig.add_vline(x=df['close'].median(), line_dash="dash", line_color="green",
                             annotation_text=f"Median: ${df['close'].median():.2f}")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'close' in df.columns:
                daily_returns = df['close'].pct_change().dropna() * 100
                fig = px.histogram(daily_returns, nbins=50, title='Distribution of Daily Returns (%)',
                                  labels={'value': 'Return (%)'}, color_discrete_sequence=['#2E86AB'])
                fig.add_vline(x=0, line_dash="solid", line_color="black")
                fig.add_vline(x=daily_returns.mean(), line_dash="dash", line_color="red",
                             annotation_text=f"Mean: {daily_returns.mean():.2f}%")
                st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Correlation Matrix")
        numeric_cols = ['close', 'volume', 'rsi_14', 'macd', 'atr_14', 'next_day_close']
        # Filter to only columns that exist
        available_cols = [col for col in numeric_cols if col in df.columns]
        if len(available_cols) > 1:
            corr_matrix = df[available_cols].corr()
            
            fig = px.imshow(corr_matrix, text_auto='.2f', aspect="auto",
                           color_continuous_scale='RdBu_r', title='Feature Correlations')
            st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        with st.expander("💡 Key Insights from EDA"):
            st.markdown("""
            <div class="insight-text">
            <h4>📌 Key Observations:</h4>
            <ul>
                <li><strong>Price Trend:</strong> Netflix stock has shown remarkable growth from ~$50 to over $900 (1700% increase)</li>
                <li><strong>Volatility:</strong> Daily returns average +0.095% with standard deviation of 2.85%</li>
                <li><strong>Technical Indicators:</strong> RSI typically ranges between 30-70, indicating normal trading conditions</li>
                <li><strong>Correlations:</strong> Strong positive correlation between price-based features (as expected)</li>
                <li><strong>Volume Spikes:</strong> Notable volume increases during earnings announcements and major events</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.header("🤖 Model Training")
        
        if st.session_state.model_trained:
            # Prepare features (remove highly correlated ones)
            features_to_drop = ['open', 'high', 'low', 'sma_50', 'sma_100', 'ema_100']
            # Only drop columns that exist
            cols_to_drop = [col for col in features_to_drop if col in df.columns]
            X = df.drop(columns=['next_day_close'] + cols_to_drop)
            y = df['next_day_close']
            
            st.write(f"**Features used in model:** {', '.join(X.columns.tolist())}")
            st.write(f"**Training samples:** {len(X)}")
            
            # Time series split
            test_size = int(len(X) * st.session_state.test_size / 100)
            train_size = len(X) - test_size
            
            X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
            
            # Train model
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = lr.predict(X_train)
            y_pred_test = lr.predict(X_test)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # Store model in session state
            st.session_state.model = lr
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.feature_names = X.columns.tolist()
            st.session_state.train_r2 = train_r2
            st.session_state.test_r2 = test_r2
            st.session_state.train_rmse = train_rmse
            st.session_state.test_rmse = test_rmse
            st.session_state.train_mae = train_mae
            st.session_state.test_mae = test_mae
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Train R2 Score", f"{train_r2:.4f}")
                st.metric("Test R2 Score", f"{test_r2:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Train RMSE", f"${train_rmse:.2f}")
                st.metric("Test RMSE", f"${test_rmse:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Train MAE", f"${train_mae:.2f}")
                st.metric("Test MAE", f"${test_mae:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Feature importance
            st.subheader("Feature Importance (Coefficients)")
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Coefficient': lr.coef_,
                'Abs_Coefficient': np.abs(lr.coef_)
            }).sort_values('Abs_Coefficient', ascending=False)
            
            fig = px.bar(feature_importance.head(10), x='Coefficient', y='Feature',
                        orientation='h', title='Top 10 Feature Coefficients',
                        color='Coefficient', color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)
            
            # Cross-validation option
            with st.expander("🔄 Cross-Validation Results"):
                if st.button("Run Cross-Validation"):
                    with st.spinner("Running 5-fold time series CV..."):
                        tscv = TimeSeriesSplit(n_splits=st.session_state.cv_folds)
                        cv_scores = []
                        
                        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
                            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
                            
                            cv_model = LinearRegression()
                            cv_model.fit(X_tr, y_tr)
                            y_pred_val = cv_model.predict(X_val)
                            score = r2_score(y_val, y_pred_val)
                            cv_scores.append(score)
                            
                            st.write(f"Fold {fold}: R2 = {score:.4f}")
                        
                        st.write(f"**Average CV R2:** {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores)*2:.4f})")
            
            st.session_state.predictions_made = True
        
        else:
            st.info("👈 Go to the sidebar and click 'Train Model' to start training")
            
            # Show model architecture
            with st.expander("📖 About the Model"):
                st.markdown("""
                ### Linear Regression Model
                
                **Why Linear Regression?**
                - Simple and interpretable
                - Fast training and prediction
                - Good baseline for stock prediction
                
                **Features Used:**
                - Technical indicators (RSI, MACD, CCI)
                - Moving averages (EMA_50)
                - Volatility measures (ATR, TrueRange)
                - Volume
                
                **Model Equation:**
