import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NFLX Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
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

# ── Session state initialisation ───────────────────────────────────────────────
DEFAULTS = {
    'data_loaded': False,
    'model_trained': False,
    'predictions_made': False,
    'df': None,
    'model': None,
    'scaler': None,
    'feature_names': [],
    'X_train': None,
    'X_test': None,
    'y_train': None,
    'y_test': None,
    'y_pred_train': None,   # FIX: store predictions so tab4 can use them
    'y_pred_test': None,
    'train_r2': None,
    'test_r2': None,
    'train_rmse': None,
    'test_rmse': None,
    'train_mae': None,
    'test_mae': None,
    'test_size': 20,        # FIX: initialise so tab3 never KeyErrors
    'remove_outliers': False,
    'scale_features': True,
    'cv_folds': 5,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">NETFLIX STOCK PREDICTOR</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Linear Regression Model for Next Day Close Price Prediction (2014-2023)</p>',
    unsafe_allow_html=True
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/7/75/Netflix_icon.svg", width=100)
    st.title("⚙️ Control Panel")

    uploaded_file = st.file_uploader(
        "Upload CSV file", type=['csv'],
        help="Upload your Netflix stock data CSV"
    )

    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            # FIX: be flexible about the date column name (date / Date / DATE)
            date_col = next((c for c in df_raw.columns if c.lower() == 'date'), None)
            if date_col is None:
                st.error("CSV must contain a 'date' column.")
            else:
                df_raw[date_col] = pd.to_datetime(df_raw[date_col])
                df_raw = df_raw.rename(columns={date_col: 'date'})
                df_raw.set_index('date', inplace=True)
                df_raw.sort_index(inplace=True)          # ensure chronological order

                # FIX: guard against missing target column
                if 'next_day_close' not in df_raw.columns:
                    st.error("CSV must contain a 'next_day_close' target column.")
                else:
                    st.session_state.df = df_raw
                    st.session_state.data_loaded = True
                    # Reset downstream state when a new file is uploaded
                    st.session_state.model_trained = False
                    st.session_state.predictions_made = False
                    st.success("✅ Data loaded successfully!")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.session_state.data_loaded = False

    if not st.session_state.data_loaded:
        st.info("📌 Please upload your CSV file to begin")

    if st.session_state.data_loaded and st.session_state.df is not None:
        st.markdown("---")
        st.subheader("🔧 Model Parameters")

        st.session_state.test_size = st.slider("Test Size (%)", 10, 30, 20, 5)

        with st.expander("Advanced Options"):
            st.session_state.remove_outliers = st.checkbox("Remove Outliers", False)
            st.session_state.scale_features = st.checkbox("Scale Features", True)
            st.session_state.cv_folds = st.slider("Cross-validation Folds", 2, 10, 5)

        if st.button("🚀 Train Model", use_container_width=True):
            st.session_state.model_trained = True
            st.session_state.predictions_made = False  # reset on retrain

# ── Main content ───────────────────────────────────────────────────────────────
if st.session_state.data_loaded and st.session_state.df is not None:
    df = st.session_state.df

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview",
        "📈 Exploratory Analysis",
        "🤖 Model Training",
        "📉 Predictions",
        "📋 Summary Report"
    ])

    # ── TAB 1 : Data Overview ──────────────────────────────────────────────────
    with tab1:
        st.header("📊 Data Overview")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Days", f"{len(df):,}")
        with col2:
            st.metric("Date Range", f"{df.index.min().year}–{df.index.max().year}")
        with col3:
            total_return = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
            st.metric("Avg Close Price", f"${df['close'].mean():.2f}",
                      delta=f"{total_return:.1f}% total return")
        with col4:
            st.metric("Total Volume", f"{df['volume'].sum():,.0f}")

        st.subheader("Data Preview")
        col_left, col_right = st.columns([3, 1])
        with col_right:
            rows_to_show = st.selectbox("Rows to display", [5, 10, 20, 50, 100], index=1)
        with col_left:
            st.dataframe(df.head(rows_to_show), use_container_width=True)

        st.subheader("Dataset Information")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Column Names:**")
            for col in df.columns:
                st.write(f"• {col}")
        with c2:
            info_df = pd.DataFrame({
                'Data Type': df.dtypes,
                'Missing': df.isnull().sum(),
                'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(info_df, use_container_width=True)

        with st.expander("📊 Statistical Summary"):
            st.dataframe(df.describe().round(2), use_container_width=True)

    # ── TAB 2 : Exploratory Analysis ──────────────────────────────────────────
    with tab2:
        st.header("📈 Exploratory Data Analysis")

        st.subheader("Stock Price Over Time")
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Close Price with Range', 'Trading Volume', 'RSI-14', 'MACD'),
            vertical_spacing=0.12
        )

        fig.add_trace(
            go.Scatter(x=df.index, y=df['close'], mode='lines', name='Close',
                       line=dict(color='#E50914', width=2)),
            row=1, col=1
        )
        if 'high' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['high'], mode='lines', name='High',
                           line=dict(color='rgba(0,200,0,0.4)', width=1), showlegend=False),
                row=1, col=1
            )
        if 'low' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['low'], mode='lines', name='Low',
                           line=dict(color='rgba(255,100,100,0.4)', width=1), showlegend=False),
                row=1, col=1
            )
        if 'volume' in df.columns:
            fig.add_trace(
                go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color='#2E86AB'),
                row=1, col=2
            )
        if 'rsi_14' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['rsi_14'], mode='lines', name='RSI-14',
                           line=dict(color='#F39C12', width=2)),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
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

        st.subheader("Distribution Analysis")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x='close', nbins=50, title='Distribution of Close Price',
                               labels={'close': 'Price ($)'}, color_discrete_sequence=['#E50914'])
            fig.add_vline(x=df['close'].mean(), line_dash="dash", line_color="blue",
                          annotation_text=f"Mean: ${df['close'].mean():.2f}")
            fig.add_vline(x=df['close'].median(), line_dash="dash", line_color="green",
                          annotation_text=f"Median: ${df['close'].median():.2f}")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            daily_returns = df['close'].pct_change().dropna() * 100
            fig = px.histogram(daily_returns, nbins=50, title='Distribution of Daily Returns (%)',
                               labels={'value': 'Return (%)'}, color_discrete_sequence=['#2E86AB'])
            fig.add_vline(x=0, line_dash="solid", line_color="black")
            fig.add_vline(x=daily_returns.mean(), line_dash="dash", line_color="red",
                          annotation_text=f"Mean: {daily_returns.mean():.3f}%")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Correlation Matrix")
        candidate_cols = ['close', 'volume', 'rsi_14', 'macd', 'atr_14', 'next_day_close']
        available_cols = [c for c in candidate_cols if c in df.columns]
        if len(available_cols) > 1:
            corr_matrix = df[available_cols].corr()
            fig = px.imshow(corr_matrix, text_auto='.2f', aspect="auto",
                            color_continuous_scale='RdBu_r', title='Feature Correlations')
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("💡 Key Insights from EDA"):
            st.markdown("""
            <div class="insight-text">
            <h4>📌 Key Observations:</h4>
            <ul>
                <li><strong>Price Trend:</strong> Netflix stock has shown remarkable growth from ~$50 to over $900 (~1700% increase)</li>
                <li><strong>Volatility:</strong> Daily returns average ~+0.095% with std deviation of ~2.85%</li>
                <li><strong>Technical Indicators:</strong> RSI typically ranges 30–70, indicating normal trading conditions</li>
                <li><strong>Correlations:</strong> Strong positive correlation between price-based features (as expected)</li>
                <li><strong>Volume Spikes:</strong> Notable volume increases during earnings announcements and major events</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 3 : Model Training ─────────────────────────────────────────────────
    with tab3:
        st.header("🤖 Model Training")

        if st.session_state.model_trained:
            # ── Feature preparation ────────────────────────────────────────────
            COLS_TO_DROP = ['open', 'high', 'low', 'sma_50', 'sma_100', 'ema_100']
            drop_cols = [c for c in COLS_TO_DROP if c in df.columns]

            # FIX: drop target + unwanted cols, guard against missing columns
            X = df.drop(columns=['next_day_close'] + drop_cols)
            y = df['next_day_close']

            # FIX: remove_outliers is now actually applied
            if st.session_state.remove_outliers:
                z_scores = np.abs((X - X.mean()) / X.std())
                mask = (z_scores < 3).all(axis=1)
                X, y = X[mask], y[mask]
                st.info(f"Outlier removal: kept {mask.sum():,} / {len(mask):,} rows.")

            # Drop rows where any NaN exists (technical indicators create NaNs at start)
            valid_mask = X.notna().all(axis=1) & y.notna()
            X, y = X[valid_mask], y[valid_mask]

            test_size_n = int(len(X) * st.session_state.test_size / 100)
            train_size_n = len(X) - test_size_n

            X_train = X.iloc[:train_size_n]
            X_test  = X.iloc[train_size_n:]
            y_train = y.iloc[:train_size_n]
            y_test  = y.iloc[train_size_n:]

            st.write(f"**Features used:** {', '.join(X.columns.tolist())}")
            st.write(f"**Train samples:** {len(X_train):,} | **Test samples:** {len(X_test):,}")

            # FIX: scale_features now actually applied via Pipeline
            if st.session_state.scale_features:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('lr', LinearRegression())
                ])
            else:
                pipeline = Pipeline([('lr', LinearRegression())])

            pipeline.fit(X_train, y_train)

            y_pred_train = pipeline.predict(X_train)
            y_pred_test  = pipeline.predict(X_test)

            train_r2   = r2_score(y_train, y_pred_train)
            test_r2    = r2_score(y_test,  y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse  = np.sqrt(mean_squared_error(y_test,  y_pred_test))
            train_mae  = mean_absolute_error(y_train, y_pred_train)
            test_mae   = mean_absolute_error(y_test,  y_pred_test)

            # Persist to session state
            st.session_state.model        = pipeline
            st.session_state.X_train      = X_train
            st.session_state.X_test       = X_test
            st.session_state.y_train      = y_train
            st.session_state.y_test       = y_test
            st.session_state.y_pred_train = y_pred_train   # FIX: stored
            st.session_state.y_pred_test  = y_pred_test    # FIX: stored
            st.session_state.feature_names= X.columns.tolist()
            st.session_state.train_r2     = train_r2
            st.session_state.test_r2      = test_r2
            st.session_state.train_rmse   = train_rmse
            st.session_state.test_rmse    = test_rmse
            st.session_state.train_mae    = train_mae
            st.session_state.test_mae     = test_mae
            st.session_state.predictions_made = True  # FIX: set here after training

            # ── Metrics ───────────────────────────────────────────────────────
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Train R²", f"{train_r2:.4f}")
                st.metric("Test R²",  f"{test_r2:.4f}")
            with c2:
                st.metric("Train RMSE", f"${train_rmse:.2f}")
                st.metric("Test RMSE",  f"${test_rmse:.2f}")
            with c3:
                st.metric("Train MAE", f"${train_mae:.2f}")
                st.metric("Test MAE",  f"${test_mae:.2f}")

            # ── Feature importance ────────────────────────────────────────────
            st.subheader("Feature Importance (Coefficients)")
            # FIX: access lr coefficients from pipeline correctly
            lr_step = pipeline.named_steps['lr']
            feat_imp = pd.DataFrame({
                'Feature': X.columns,
                'Coefficient': lr_step.coef_,
                'Abs_Coefficient': np.abs(lr_step.coef_)
            }).sort_values('Abs_Coefficient', ascending=False)

            fig = px.bar(feat_imp.head(10), x='Coefficient', y='Feature',
                         orientation='h', title='Top 10 Feature Coefficients',
                         color='Coefficient', color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)

            # ── Cross-validation ──────────────────────────────────────────────
            with st.expander("🔄 Cross-Validation Results"):
                if st.button("Run Cross-Validation"):
                    with st.spinner(f"Running {st.session_state.cv_folds}-fold time series CV…"):
                        tscv = TimeSeriesSplit(n_splits=st.session_state.cv_folds)
                        cv_scores = []
                        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
                            cv_pipe = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
                            cv_pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
                            score = r2_score(y.iloc[val_idx], cv_pipe.predict(X.iloc[val_idx]))
                            cv_scores.append(score)
                            st.write(f"Fold {fold}: R² = {score:.4f}")
                        st.write(f"**Average CV R²:** {np.mean(cv_scores):.4f}"
                                 f" (±{np.std(cv_scores)*2:.4f})")

        else:
            st.info("👈 Go to the sidebar and click **Train Model** to start training.")

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

                **Target:** `next_day_close` — next trading day's closing price.

                **Equation:**
                `ŷ = β₀ + β₁x₁ + β₂x₂ + … + βₙxₙ`
                """)

    # ── TAB 4 : Predictions ────────────────────────────────────────────────────
    with tab4:
        st.header("📉 Predictions")

        if st.session_state.predictions_made:
            y_test       = st.session_state.y_test
            y_pred_test  = st.session_state.y_pred_test
            y_train      = st.session_state.y_train
            y_pred_train = st.session_state.y_pred_train
            X_test       = st.session_state.X_test

            # ── Actual vs Predicted ───────────────────────────────────────────
            st.subheader("Actual vs Predicted — Test Set")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=X_test.index, y=y_test,
                mode='lines', name='Actual',
                line=dict(color='#2E86AB', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=X_test.index, y=y_pred_test,
                mode='lines', name='Predicted',
                line=dict(color='#E50914', width=2, dash='dash')
            ))
            fig.update_layout(
                title='Actual vs Predicted Close Price (Test Period)',
                xaxis_title='Date', yaxis_title='Price ($)',
                hovermode='x unified', height=450
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Scatter: Actual vs Predicted ──────────────────────────────────
            st.subheader("Prediction Scatter Plot")
            c1, c2 = st.columns(2)
            with c1:
                fig = px.scatter(
                    x=y_test, y=y_pred_test,
                    labels={'x': 'Actual Price ($)', 'y': 'Predicted Price ($)'},
                    title='Actual vs Predicted (Test Set)',
                    color_discrete_sequence=['#E50914'],
                    opacity=0.6
                )
                min_val = min(y_test.min(), y_pred_test.min())
                max_val = max(y_test.max(), y_pred_test.max())
                fig.add_shape(type='line', x0=min_val, y0=min_val,
                              x1=max_val, y1=max_val,
                              line=dict(color='black', dash='dash'))
                st.plotly_chart(fig, use_container_width=True)

            # ── Residuals ─────────────────────────────────────────────────────
            with c2:
                residuals = y_test - y_pred_test
                fig = px.histogram(residuals, nbins=40, title='Residual Distribution (Test Set)',
                                   labels={'value': 'Residual ($)'},
                                   color_discrete_sequence=['#2E86AB'])
                fig.add_vline(x=0, line_dash='dash', line_color='red')
                fig.add_vline(x=residuals.mean(), line_dash='dot', line_color='orange',
                              annotation_text=f"Mean: ${residuals.mean():.2f}")
                st.plotly_chart(fig, use_container_width=True)

            # ── Residuals over time ───────────────────────────────────────────
            st.subheader("Residuals Over Time")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=X_test.index, y=residuals,
                mode='lines+markers', name='Residual',
                line=dict(color='#8E44AD', width=1),
                marker=dict(size=3)
            ))
            fig.add_hline(y=0, line_dash='dash', line_color='black', opacity=0.5)
            fig.update_layout(
                title='Residuals Over Time (Test Period)',
                xaxis_title='Date', yaxis_title='Residual ($)',
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Prediction error metrics table ────────────────────────────────
            st.subheader("Detailed Prediction Error Analysis")
            error_pct = np.abs(residuals / y_test * 100)
            error_df = pd.DataFrame({
                'Actual':       y_test.values,
                'Predicted':    y_pred_test,
                'Residual':     residuals.values,
                'Error %':      error_pct.values
            }, index=X_test.index).round(2)
            st.dataframe(error_df.tail(20), use_container_width=True)
            st.caption(f"Mean Absolute Percentage Error (MAPE): {error_pct.mean():.2f}%")

        else:
            st.info("👈 Train the model first (sidebar → **Train Model**).")

    # ── TAB 5 : Summary Report ─────────────────────────────────────────────────
    with tab5:
        st.header("📋 Summary Report")

        if st.session_state.predictions_made:
            # ── Dataset summary ───────────────────────────────────────────────
            st.subheader("1. Dataset Summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Records", f"{len(df):,}")
            c2.metric("Date Range", f"{df.index.min().date()} → {df.index.max().date()}")
            c3.metric("Features Used", len(st.session_state.feature_names))

            # ── Model performance ─────────────────────────────────────────────
            st.subheader("2. Model Performance")
            perf_df = pd.DataFrame({
                'Metric': ['R² Score', 'RMSE ($)', 'MAE ($)'],
                'Train':  [f"{st.session_state.train_r2:.4f}",
                           f"${st.session_state.train_rmse:.2f}",
                           f"${st.session_state.train_mae:.2f}"],
                'Test':   [f"{st.session_state.test_r2:.4f}",
                           f"${st.session_state.test_rmse:.2f}",
                           f"${st.session_state.test_mae:.2f}"]
            })
            st.table(perf_df)

            overfit_gap = st.session_state.train_r2 - st.session_state.test_r2
            if overfit_gap > 0.05:
                st.warning(f"⚠️ Potential overfitting detected (Train R² − Test R² = {overfit_gap:.4f}).")
            else:
                st.success("✅ Train/Test R² gap is within acceptable range — model generalises well.")

            # ── Feature importance summary ────────────────────────────────────
            st.subheader("3. Top Features by Importance")
            lr_step = st.session_state.model.named_steps['lr']
            feat_imp = pd.DataFrame({
                'Feature':    st.session_state.feature_names,
                'Coefficient': lr_step.coef_,
                '|Coefficient|': np.abs(lr_step.coef_)
            }).sort_values('|Coefficient|', ascending=False).head(10).reset_index(drop=True)
            st.dataframe(feat_imp.round(4), use_container_width=True)

            # ── Conclusion ────────────────────────────────────────────────────
            st.subheader("4. Conclusions & Recommendations")
            st.markdown("""
            <div class="insight-text">
            <h4>📌 Model Summary</h4>
            <ul>
                <li>A <strong>Linear Regression</strong> pipeline (with optional StandardScaler) was trained on chronologically split data.</li>
                <li>Time-based splitting prevents look-ahead bias — a critical requirement for financial data.</li>
                <li>Test RMSE and MAE give the expected dollar-range prediction error on unseen data.</li>
            </ul>
            <h4>⚠️ Limitations</h4>
            <ul>
                <li>Linear Regression assumes a linear relationship; stock prices are driven by non-linear dynamics.</li>
                <li>Past technical indicators do not guarantee future price direction.</li>
                <li>This model is for <strong>educational purposes only</strong> — not investment advice.</li>
            </ul>
            <h4>🚀 Next Steps</h4>
            <ul>
                <li>Try non-linear models (XGBoost, LSTM) for potentially better accuracy.</li>
                <li>Add macro-economic features (interest rates, VIX) as external regressors.</li>
                <li>Use walk-forward (rolling window) validation for more robust performance estimates.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.info("👈 Train the model first to generate the summary report.")

else:
    # Landing page when no data is loaded
    st.markdown("""
    ## 👋 Welcome to the Netflix Stock Predictor!

    **Get started in 3 steps:**
    1. **Upload** your Netflix stock CSV via the sidebar.
    2. **Adjust** model parameters (test size, scaling, CV folds).
    3. **Click Train Model** and explore the results across all tabs.

    > The CSV must include a `date` column and a `next_day_close` target column.
    """)
