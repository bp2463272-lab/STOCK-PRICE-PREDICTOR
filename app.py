
**Evaluation Metrics:**
- R² Score (0-1, higher is better)
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
""")

with tab4:
st.header("📉 Predictions & Analysis")

if st.session_state.get('predictions_made', False) and st.session_state.model is not None:
model = st.session_state.model
X_test = st.session_state.X_test
y_test = st.session_state.y_test
y_pred = model.predict(X_test)

# Create predictions dataframe
pred_df = pd.DataFrame({
'Actual': y_test.values,
'Predicted': y_pred,
'Date': y_test.index
})
pred_df['Error'] = pred_df['Actual'] - pred_df['Predicted']
pred_df['Error %'] = (pred_df['Error'] / pred_df['Actual'] * 100).abs()

# Actual vs Predicted plot
st.subheader("Actual vs Predicted Prices")
fig = make_subplots(rows=1, cols=2, subplot_titles=('Time Series', 'Scatter Plot'))

# Time series
fig.add_trace(
go.Scatter(x=pred_df['Date'], y=pred_df['Actual'], mode='lines',
          name='Actual', line=dict(color='#E50914', width=2)),
row=1, col=1
)
fig.add_trace(
go.Scatter(x=pred_df['Date'], y=pred_df['Predicted'], mode='lines',
          name='Predicted', line=dict(color='#2E86AB', width=2, dash='dash')),
row=1, col=1
)

# Scatter
fig.add_trace(
go.Scatter(x=pred_df['Actual'], y=pred_df['Predicted'], mode='markers',
          marker=dict(color='#F39C12', size=5, opacity=0.6),
          name='Predictions'),
row=1, col=2
)

# Perfect prediction line
min_val = min(pred_df['Actual'].min(), pred_df['Predicted'].min())
max_val = max(pred_df['Actual'].max(), pred_df['Predicted'].max())
fig.add_trace(
go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
          mode='lines', line=dict(color='red', dash='dash'),
          name='Perfect Fit', showlegend=False),
row=1, col=2
)

fig.update_layout(height=500)
st.plotly_chart(fig, use_container_width=True)

# Error analysis
col1, col2, col3 = st.columns(3)
with col1:
st.metric("Mean Error", f"${pred_df['Error'].mean():.2f}")
with col2:
st.metric("Std Error", f"${pred_df['Error'].std():.2f}")
with col3:
st.metric("Avg Error %", f"{pred_df['Error %'].mean():.2f}%")

# Residual plot
st.subheader("Residual Analysis")
fig = make_subplots(rows=1, cols=2, subplot_titles=('Residuals vs Predicted', 'Residual Distribution'))

fig.add_trace(
go.Scatter(x=y_pred, y=pred_df['Error'], mode='markers',
          marker=dict(color='#8E44AD', size=5, opacity=0.6),
          name='Residuals'),
row=1, col=1
)
fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

fig.add_trace(
go.Histogram(x=pred_df['Error'], nbinsx=30, name='Residuals',
            marker_color='#2A9D8F'),
row=1, col=2
)

fig.update_layout(height=400)
st.plotly_chart(fig, use_container_width=True)

# Make new prediction
st.subheader("🔮 Make New Prediction")
st.write("Enter feature values to predict next day's close price:")

col1, col2, col3 = st.columns(3)
input_values = {}

feature_names = st.session_state.feature_names
n_features = len(feature_names)

with col1:
for feat in feature_names[:min(4, n_features)]:
    if feat in X_test.columns:
        default_val = float(X_test[feat].mean())
        input_values[feat] = st.number_input(f"{feat}", value=default_val, format="%.2f", key=f"input_{feat}")

with col2:
for feat in feature_names[4:8] if n_features > 4 else []:
    if feat in X_test.columns:
        default_val = float(X_test[feat].mean())
        input_values[feat] = st.number_input(f"{feat}", value=default_val, format="%.2f", key=f"input_{feat}")

with col3:
for feat in feature_names[8:] if n_features > 8 else []:
    if feat in X_test.columns:
        default_val = float(X_test[feat].mean())
        input_values[feat] = st.number_input(f"{feat}", value=default_val, format="%.2f", key=f"input_{feat}")

if st.button("Predict Next Day Close"):
input_df = pd.DataFrame([input_values])
# Ensure columns are in the same order as training
input_df = input_df[feature_names]
prediction = model.predict(input_df)[0]

# Get confidence interval (simplified)
std_error = pred_df['Error'].std()

st.success(f"### Predicted Next Day Close: **${prediction:.2f}**")
st.info(f"Confidence Interval (95%): ${prediction - 1.96*std_error:.2f} to ${prediction + 1.96*std_error:.2f}")

else:
st.info("👈 Train the model first to see predictions")

with tab5:
st.header("📋 Summary Report")

if st.session_state.get('predictions_made', False) and st.session_state.model is not None:
# Generate comprehensive report
feature_importance = pd.DataFrame({
'Feature': st.session_state.feature_names,
'Coefficient': st.session_state.model.coef_,
'Abs_Coefficient': np.abs(st.session_state.model.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

# Get prediction data for error analysis
X_test = st.session_state.X_test
y_test = st.session_state.y_test
y_pred = st.session_state.model.predict(X_test)
pred_df = pd.DataFrame({
'Actual': y_test.values,
'Predicted': y_pred,
'Error': y_test.values - y_pred
})

report = f"""
## Linear Regression Model Summary

### 1. Data Overview
- **Total Trading Days:** {len(df):,}
- **Date Range:** {df.index.min().date()} to {df.index.max().date()}
- **Features Used:** {len(st.session_state.feature_names)}
- **Target Variable:** next_day_close (Next Day's Closing Price)

### 2. Model Performance
- **Train R² Score:** {st.session_state.train_r2:.4f} ({st.session_state.train_r2*100:.2f}% variance explained)
- **Test R² Score:** {st.session_state.test_r2:.4f} ({st.session_state.test_r2*100:.2f}% variance explained)
- **Train RMSE:** ${st.session_state.train_rmse:.2f}
- **Test RMSE:** ${st.session_state.test_rmse:.2f}
- **Train MAE:** ${st.session_state.train_mae:.2f}
- **Test MAE:** ${st.session_state.test_mae:.2f}

### 3. Top 5 Most Important Features
"""

for i, row in feature_importance.head(5).iterrows():
report += f"\n   - **{row['Feature']}:** Coefficient = {row['Coefficient']:.4f}"

report += f"""

### 4. Error Analysis
- **Mean Prediction Error:** ${pred_df['Error'].mean():.2f}
- **Standard Deviation of Error:** ${pred_df['Error'].std():.2f}
- **Average Absolute Error %:** {(pred_df['Error'].abs() / pred_df['Actual'] * 100).mean():.2f}%
- **Maximum Overestimate:** ${pred_df['Error'].max():.2f}
- **Maximum Underestimate:** ${-pred_df['Error'].min():.2f}

### 5. Business Implications
- The model explains **{st.session_state.test_r2*100:.1f}%** of the variance in next day's closing price
- Typical prediction error is **${st.session_state.test_mae:.2f}** (about {st.session_state.test_mae/df['close'].mean()*100:.1f}% of average price)
- Most influential features are moving averages and volatility indicators
- The model performs best during normal market conditions

### 6. Limitations & Recommendations
**Limitations:**
- Linear model cannot capture complex non-linear patterns
- Does not account for external factors (news, earnings, market sentiment)
- Performance may degrade during high volatility periods

**Recommendations for Improvement:**
1. Try non-linear models (Random Forest, XGBoost)
2. Add external features (market indices, sector performance)
3. Implement ensemble methods
4. Use LSTM for better time series pattern recognition
"""

st.markdown(report)

# Download report button
report_df = pd.DataFrame({
'Metric': ['Train R²', 'Test R²', 'Train RMSE', 'Test RMSE', 'Train MAE', 'Test MAE'],
'Value': [st.session_state.train_r2, st.session_state.test_r2, 
         st.session_state.train_rmse, st.session_state.test_rmse,
         st.session_state.train_mae, st.session_state.test_mae]
})

csv = report_df.to_csv(index=False)
st.download_button(
label="📥 Download Report CSV",
data=csv,
file_name="model_report.csv",
mime="text/csv"
)

else:
st.info("👈 Train the model first to generate a summary report")

else:
# Welcome message when no data is loaded
st.info("👈 Please upload a CSV file in the sidebar to get started")

# Show sample of expected format
st.markdown("""
### Expected CSV Format:

Your CSV should contain the following columns:
- `date`: Trading date
- `open`: Opening price
- `high`: Daily high price
- `low`: Daily low price
- `close`: Closing price
- `volume`: Trading volume
- Various technical indicators (RSI, MACD, etc.)
- `next_day_close`: Target variable (next day's closing price)

**Example:**
