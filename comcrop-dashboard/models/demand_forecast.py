import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
import base64
from sklearn.linear_model import LinearRegression

def generate_forecast_data(data_file, model_file=None, days_to_forecast=60):
    """
    Generate forecast data for the dashboard
    
    Args:
        data_file: CSV file path with historical sales data
        model_file: Pickle file with trained model (if None, will train a new one)
        days_to_forecast: Number of days to forecast
    
    Returns:
        tuple: (svg_code, metrics_dict)
    """
    # Load the historical sales data
    df = pd.read_csv(data_file)
    
    # Preprocess data
    df['Date Sold'] = pd.to_datetime(df['Date Sold'], format='%d/%m/%Y')
    df = df.sort_values('Date Sold')
    
    # Extract time-related features
    df['Year'] = df['Date Sold'].dt.year
    df['Month'] = df['Date Sold'].dt.month
    df['Day'] = df['Date Sold'].dt.day
    df['DayOfWeek'] = df['Date Sold'].dt.dayofweek
    
    # Group by date to get total quantity sold per day
    daily_sales = df.groupby('Date Sold')['Quantity Sold'].sum().reset_index()
    daily_sales.columns = ['Date', 'Total Quantity']
    
    # Ensure all dates are present by resampling
    date_range = pd.date_range(start=daily_sales['Date'].min(), end=daily_sales['Date'].max())
    daily_sales = daily_sales.set_index('Date').reindex(date_range).fillna(0).reset_index()
    daily_sales.columns = ['Date', 'Total Quantity']
    
    # Create time-based features
    daily_sales['Year'] = daily_sales['Date'].dt.year
    daily_sales['Month'] = daily_sales['Date'].dt.month
    daily_sales['Day'] = daily_sales['Date'].dt.day
    daily_sales['DayOfWeek'] = daily_sales['Date'].dt.dayofweek
    daily_sales['DayOfYear'] = daily_sales['Date'].dt.dayofyear
    daily_sales['WeekOfYear'] = daily_sales['Date'].dt.isocalendar().week
    
    # Create lags (previous days' sales)
    for lag in [1, 2, 3, 7, 14]:
        daily_sales[f'Lag_{lag}'] = daily_sales['Total Quantity'].shift(lag)
    
    # Add rolling statistics
    daily_sales['RollingMean_7'] = daily_sales['Total Quantity'].rolling(window=7).mean().shift(1)
    daily_sales['RollingMax_7'] = daily_sales['Total Quantity'].rolling(window=7).max().shift(1)
    daily_sales['RollingMean_14'] = daily_sales['Total Quantity'].rolling(window=14).mean().shift(1)
    
    # Drop NaN values which result from the lag/rolling features
    daily_sales = daily_sales.dropna()
    
    # Define features for the model
    feature_columns = ['Month', 'Day', 'DayOfWeek', 'DayOfYear', 'WeekOfYear', 
                       'Lag_1', 'Lag_2', 'Lag_3', 'Lag_7', 'Lag_14',
                       'RollingMean_7', 'RollingMax_7', 'RollingMean_14']
    
    X = daily_sales[feature_columns]
    y = daily_sales['Total Quantity']
    
    # If model file is provided, load the model, otherwise train a new one
    if model_file and os.path.exists(model_file):
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
    else:
        # Train a simple linear regression model
        model = LinearRegression()
        model.fit(X, y)
    
    # Make predictions for the historical data
    daily_sales['Predicted'] = model.predict(X)
    
    # Generate future dates for forecasting
    last_date = daily_sales['Date'].max()
    
    # Function to generate future features and predictions
    def generate_future_features(last_date, num_days, last_n_rows):
        # Create future dates
        future_dates = [last_date + timedelta(days=i+1) for i in range(num_days)]
        
        # Initialize DataFrame with date column
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Total Quantity': [np.nan] * num_days  # Placeholder for predictions
        })
        
        # Add time-based features
        future_df['Year'] = future_df['Date'].dt.year
        future_df['Month'] = future_df['Date'].dt.month
        future_df['Day'] = future_df['Date'].dt.day
        future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek
        future_df['DayOfYear'] = future_df['Date'].dt.dayofyear
        future_df['WeekOfYear'] = future_df['Date'].dt.isocalendar().week
        
        # Start with the last known values for lagged features
        # Get the last N known rows
        last_rows = last_n_rows.copy()
        
        # Generate predictions day by day
        for i in range(num_days):
            # Set the lag values based on previous data
            future_df.loc[i, 'Lag_1'] = last_rows['Total Quantity'].iloc[-1]
            future_df.loc[i, 'Lag_2'] = last_rows['Total Quantity'].iloc[-2] if len(last_rows) > 1 else last_rows['Total Quantity'].iloc[-1]
            future_df.loc[i, 'Lag_3'] = last_rows['Total Quantity'].iloc[-3] if len(last_rows) > 2 else last_rows['Total Quantity'].iloc[-1]
            future_df.loc[i, 'Lag_7'] = last_rows['Total Quantity'].iloc[-7] if len(last_rows) > 6 else last_rows['Total Quantity'].iloc[-1]
            future_df.loc[i, 'Lag_14'] = last_rows['Total Quantity'].iloc[-14] if len(last_rows) > 13 else last_rows['Total Quantity'].iloc[-1]
            
            # Set rolling statistics
            future_df.loc[i, 'RollingMean_7'] = last_rows['Total Quantity'].iloc[-7:].mean() if len(last_rows) >= 7 else last_rows['Total Quantity'].mean()
            future_df.loc[i, 'RollingMax_7'] = last_rows['Total Quantity'].iloc[-7:].max() if len(last_rows) >= 7 else last_rows['Total Quantity'].max()
            future_df.loc[i, 'RollingMean_14'] = last_rows['Total Quantity'].iloc[-14:].mean() if len(last_rows) >= 14 else last_rows['Total Quantity'].mean()
            
            # Make prediction for the current future day
            X_future = future_df.iloc[i:i+1][feature_columns]
            future_df.loc[i, 'Total Quantity'] = model.predict(X_future)[0]
            
            # Add the prediction to the last_rows to use for the next day's prediction
            new_row = future_df.iloc[i:i+1].copy()
            last_rows = pd.concat([last_rows, new_row])
        
        future_df['Predicted'] = future_df['Total Quantity']
        return future_df
    
    # Get the last 30 days of actual data for initial lag features
    last_n_rows = daily_sales.iloc[-30:]
    
    # Generate future predictions
    future_predictions = generate_future_features(last_date, days_to_forecast, last_n_rows)
    
    # Create a visualization
    fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
    
    # Plot historical data - show last 60 days for better visualization
    history_to_show = min(60, len(daily_sales))
    historical_data = daily_sales.iloc[-history_to_show:]
    
    # Plot actual historical values
    ax.plot(historical_data['Date'], historical_data['Total Quantity'], 
            color='#3498db', label='Actual Sales', linewidth=2)
    
    # Plot predictions for historical period
    ax.plot(historical_data['Date'], historical_data['Predicted'], 
            color='#e74c3c', linestyle='--', label='Predicted Sales', linewidth=1.5)
    
    # Plot future predictions
    ax.plot(future_predictions['Date'], future_predictions['Predicted'], 
            color='#e74c3c', linewidth=2, label='Sales Forecast')
    
    # Fill area under future predictions
    ax.fill_between(future_predictions['Date'], 0, future_predictions['Predicted'], 
                    color='#e74c3c', alpha=0.2)
    
    # Add vertical line at forecast start
    ax.axvline(x=last_date, color='#2c3e50', linestyle='--', alpha=0.7)
    ax.text(last_date, ax.get_ylim()[1]*0.95, 'Forecast Start', 
            rotation=90, verticalalignment='top', color='#2c3e50')
    
    # Format the plot
    ax.set_title('Daily Sales Forecast', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12, labelpad=10)
    ax.set_ylabel('Total Quantity Sold', fontsize=12, labelpad=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    # Format dates on x-axis
    plt.xticks(rotation=45)
    fig.tight_layout()
    
    # Convert the plot to SVG string
    buf = io.BytesIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    buf.seek(0)
    svg_code = buf.getvalue().decode('utf-8')
    plt.close(fig)
    
    # Calculate metrics for the dashboard
    # 1. Total forecasted demand for next week and month
    next_week_demand = int(future_predictions.iloc[:7]['Predicted'].sum())
    next_month_demand = int(future_predictions['Predicted'].sum())
    
    # 2. Average daily demand
    avg_daily_demand = int(future_predictions['Predicted'].mean())
    
    # 3. Peak demand day
    peak_day_idx = future_predictions['Predicted'].idxmax()
    peak_day = future_predictions.loc[peak_day_idx, 'Date'].strftime('%d %b')
    peak_demand = int(future_predictions.loc[peak_day_idx, 'Predicted'])
    
    # 4. Compare forecast to previous period
    prev_period_demand = int(daily_sales.iloc[-days_to_forecast:]['Total Quantity'].sum())
    demand_change_pct = ((next_month_demand - prev_period_demand) / prev_period_demand) * 100
    
    # 5. Calculate product breakdown for the last month
    product_sales = df.groupby('Name')['Quantity Sold'].sum().sort_values(ascending=False)
    top_products = product_sales.head(5).to_dict()
    
    # Create product breakdown SVG
    fig2, ax2 = plt.figure(figsize=(8, 5)), plt.gca()
    
    # Plot top 5 products by sales volume
    products = list(top_products.keys())
    values = list(top_products.values())
    
    colors = ['#3498db', '#2ecc71', '#f1c40f', '#e74c3c', '#9b59b6']
    ax2.bar(products, values, color=colors)
    
    ax2.set_title('Top Products by Sales Volume', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Units Sold', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Adjust layout
    fig2.tight_layout()
    
    # Convert to SVG
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='svg', bbox_inches='tight')
    buf2.seek(0)
    product_svg = buf2.getvalue().decode('utf-8')
    plt.close(fig2)
    
    # Create metrics dictionary
    metrics = {
        "next_week_demand": f"{next_week_demand} units",
        "next_month_demand": f"{next_month_demand} units",
        "avg_daily_demand": f"{avg_daily_demand} units/day",
        "peak_day": f"{peak_day} ({peak_demand} units)",
        "demand_change": f"{demand_change_pct:.1f}%",
        "top_product": f"{products[0]} ({values[0]} units)",
        "product_svg": product_svg
    }
    
    return svg_code, metrics

def get_forecast_by_product(data_file, days_to_forecast=30):
    """
    Generate forecast data for each product
    
    Args:
        data_file: CSV file path with historical sales data
        days_to_forecast: Number of days to forecast
    
    Returns:
        str: SVG visualization of product-specific forecasts
    """
    # Load the historical sales data
    df = pd.read_csv(data_file)
    
    # Preprocess data
    df['Date Sold'] = pd.to_datetime(df['Date Sold'], format='%d/%m/%Y')
    df = df.sort_values('Date Sold')
    
    # Filter data from 2024 onward
    df = df[df['Date Sold'] >= '2024-01-01']
    
    # Get top 4 products by sales volume
    top_products = df.groupby('Name')['Quantity Sold'].sum().sort_values(ascending=False).head(4).index.tolist()
    
    # Create a figure with subplots for each top product
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # For each top product, create a simple forecast
    for i, product in enumerate(top_products):
        ax = axes[i]
        
        # Filter data for this product
        product_data = df[df['Name'] == product]
        
        # Group by date
        product_daily = product_data.groupby('Date Sold')['Quantity Sold'].sum().reset_index()
        product_daily.columns = ['Date', 'Quantity']
        
        # Ensure all dates are present
        date_range = pd.date_range(start=df['Date Sold'].min(), end=df['Date Sold'].max())
        product_daily = product_daily.set_index('Date').reindex(date_range).fillna(0).reset_index()
        product_daily.columns = ['Date', 'Quantity']
        
        # Create a simple 7-day moving average forecast
        product_daily['MA7'] = product_daily['Quantity'].rolling(window=7).mean()
        
        # Plot actual data
        ax.plot(product_daily['Date'], product_daily['Quantity'], color='#3498db', alpha=0.5, label='Actual')
        
        # Plot moving average
        ax.plot(product_daily['Date'], product_daily['MA7'], color='#e74c3c', label='7-Day MA')
        
        # Forecast next N days (simple - extend the last moving average)
        last_date = product_daily['Date'].max()
        last_ma = product_daily['MA7'].iloc[-1]
        
        future_dates = [last_date + timedelta(days=i+1) for i in range(days_to_forecast)]
        ax.plot([last_date, future_dates[-1]], [last_ma, last_ma], 
                color='#e74c3c', linestyle='--', label='Forecast')
        
        # Fill area under forecast
        ax.fill_between(future_dates, 0, last_ma, color='#e74c3c', alpha=0.2)
        
        # Add vertical line at forecast start
        ax.axvline(x=last_date, color='#2c3e50', linestyle='--', alpha=0.7)
        
        # Format the subplot
        ax.set_title(f'{product}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Units Sold')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis with fewer tick labels to avoid overcrowding
        locator = plt.MaxNLocator(nbins=5)
        ax.xaxis.set_major_locator(locator)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
        
        # Add legend only to the first subplot
        if i == 0:
            ax.legend()
    
    # Set overall title
    plt.suptitle('Product-Specific Sales Forecasts', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Convert to SVG
    buf = io.BytesIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    buf.seek(0)
    product_forecast_svg = buf.getvalue().decode('utf-8')
    plt.close(fig)
    
    return product_forecast_svg