import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

# Calculate timestamps (2 years back from today)
end_time = datetime.now(timezone.utc)
start_time = end_time - timedelta(days=2 * 365)

# Convert timestamps to ISO 8601 format for Denmark API
start_time_str = start_time.strftime("%Y-%m-%dT%H:%M")
end_time_str = end_time.strftime("%Y-%m-%dT%H:%M")

def fetch_denmark_prices(start_time_str, end_time_str):
    url = "https://api.energidataservice.dk/dataset/Elspotprices"
    params = {
        'start': start_time_str,
        'end': end_time_str,
        'filter': '{"PriceArea": ["DK1"]}',
        'columns': 'HourUTC,SpotPriceEUR',
        'sort': 'HourUTC',
        'timezone': 'UTC'
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data['records'])
        
        # Ensure timestamp is in datetime format
        df['timestamp'] = pd.to_datetime(df['HourUTC'], utc=True)
        
        # Round timestamps to 3-hour intervals
        df['timestamp'] = df['timestamp'].dt.floor('3h')
        
        # Convert price from EUR/MWh to USD/kWh
        df['denmark_price_usd_kwh'] = df['SpotPriceEUR'] / 1000 * 1.08
        
        # Keep only relevant columns
        df = df[['timestamp', 'denmark_price_usd_kwh']]
        return df
    else:
        print("Error fetching Denmark prices:", response.status_code, response.text)
        return pd.DataFrame()

def fetch_texas_prices():
    """
    Generate Texas electricity prices based on ERCOT patterns
    """
    print("Using enhanced historical ERCOT price patterns")
    timestamps = pd.date_range(start=start_time, end=end_time, freq='3h', tz='UTC')
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'texas_price_usd_kwh': 0.055  # Base wholesale price
    })
    
    # Add time-based factors based on actual ERCOT patterns
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.weekday
    
    # Summer peak pricing (more nuanced)
    df['summer_factor'] = 1.0
    extreme_summer_mask = df['month'].isin([7, 8])  # July-August
    moderate_summer_mask = df['month'].isin([6, 9])  # June, September
    df.loc[extreme_summer_mask, 'summer_factor'] = 1.8
    df.loc[moderate_summer_mask, 'summer_factor'] = 1.4
    
    # Time of day pricing (more granular)
    df['time_factor'] = 1.0
    peak_mask = df['hour'].isin([14, 15, 16, 17, 18])  # 2 PM - 6 PM
    shoulder_mask = df['hour'].isin([10, 11, 12, 13, 19, 20])  # 10 AM - 1 PM, 7 PM - 8 PM
    overnight_mask = df['hour'].isin(range(1, 6))  # 1 AM - 5 AM
    
    df.loc[peak_mask, 'time_factor'] = 1.5
    df.loc[shoulder_mask, 'time_factor'] = 1.2
    df.loc[overnight_mask, 'time_factor'] = 0.7
    
    # Weekend adjustment
    df['weekend_factor'] = 1.0
    weekend_mask = df['weekday'].isin([5, 6])
    df.loc[weekend_mask, 'weekend_factor'] = 0.85
    
    # Apply all factors
    df['texas_price_usd_kwh'] = df['texas_price_usd_kwh'] * df['summer_factor'] * df['time_factor'] * df['weekend_factor']
    
    # Add controlled random variation
    np.random.seed(42)
    df['texas_price_usd_kwh'] *= (1 + np.random.normal(0, 0.03, len(df)))  # 3% variation
    
    # Ensure prices stay within historical bounds
    df['texas_price_usd_kwh'] = df['texas_price_usd_kwh'].clip(0.035, 0.18)
    
    return df[['timestamp', 'texas_price_usd_kwh']]

def fetch_kazakhstan_prices():
    """
    Generate Kazakhstan electricity prices based on historical patterns.
    Kazakhstan has relatively stable, state-regulated electricity prices with
    regional variations and slight seasonal changes.
    """
    print("Generating Kazakhstan price patterns...")
    timestamps = pd.date_range(start=start_time, end=end_time, freq='3h', tz='UTC')
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'kazakhstan_price_usd_kwh': 0.041  # Base industrial rate for mining regions
    })
    
    # Add time-based factors
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    
    # Slight seasonal variation (winter rates slightly higher due to heating demand)
    winter_mask = df['month'].isin([12, 1, 2])
    df.loc[winter_mask, 'kazakhstan_price_usd_kwh'] *= 1.15
    
    # Very minor time-of-day variation (less volatile than Western markets)
    peak_mask = df['hour'].isin([9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    df.loc[peak_mask, 'kazakhstan_price_usd_kwh'] *= 1.1
    
    # Add small random variation (Kazakhstan prices are quite stable)
    np.random.seed(42)
    df['kazakhstan_price_usd_kwh'] *= (1 + np.random.normal(0, 0.02, len(df)))
    
    # Ensure prices stay within historical bounds
    df['kazakhstan_price_usd_kwh'] = df['kazakhstan_price_usd_kwh'].clip(0.035, 0.06)
    
    return df[['timestamp', 'kazakhstan_price_usd_kwh']]

def fetch_china_prices():
    """
    Generate Chinese electricity prices based on historical patterns.
    China has regional variation and different rates for crypto mining when it was allowed.
    This models historical prices in mining-heavy regions like Sichuan and Xinjiang.
    """
    print("Generating China price patterns...")
    timestamps = pd.date_range(start=start_time, end=end_time, freq='3h', tz='UTC')
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'china_price_usd_kwh': 0.049  # Base industrial rate in mining regions
    })
    
    # Add time-based factors
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.weekday
    
    # Seasonal patterns
    # Sichuan hydropower season (abundant cheap power)
    hydro_season_mask = df['month'].isin([5, 6, 7, 8, 9])
    df.loc[hydro_season_mask, 'china_price_usd_kwh'] *= 0.7
    
    # Winter pricing (higher due to heating demand)
    winter_mask = df['month'].isin([12, 1, 2])
    df.loc[winter_mask, 'china_price_usd_kwh'] *= 1.3
    
    # Time of day pricing
    peak_mask = df['hour'].isin([10, 11, 12, 13, 14, 15, 16])
    shoulder_mask = df['hour'].isin([7, 8, 9, 17, 18, 19])
    df.loc[peak_mask, 'china_price_usd_kwh'] *= 1.2
    df.loc[shoulder_mask, 'china_price_usd_kwh'] *= 1.1
    
    # Weekend rates
    weekend_mask = df['weekday'].isin([5, 6])
    df.loc[weekend_mask, 'china_price_usd_kwh'] *= 0.9
    
    # Add controlled random variation
    np.random.seed(42)
    df['china_price_usd_kwh'] *= (1 + np.random.normal(0, 0.03, len(df)))
    
    # Ensure prices stay within historical bounds
    df['china_price_usd_kwh'] = df['china_price_usd_kwh'].clip(0.025, 0.075)
    
    return df[['timestamp', 'china_price_usd_kwh']]

def fetch_blockchain_data():
    metrics = {
        'hash-rate': 'hash_rate',
        'market-price': 'btc_market_price_usd',
        'transaction-fees': 'transaction_fees_btc'
    }
    
    all_data = []
    
    for metric, column_name in metrics.items():
        url = f'https://api.blockchain.info/charts/{metric}?timespan=2years&format=json'
        print(f"Fetching {url}")
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'values' in data:
                df = pd.DataFrame(data['values'])
                df.columns = ['timestamp', column_name]
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
                df = df.set_index('timestamp')
                
                # Resample to 3-hour intervals and interpolate
                df = df.resample('3h').mean()
                df = df.interpolate(method='linear')
                
                # If this is the fees data, calculate the diff after resampling
                if column_name == 'transaction_fees_btc':
                    df[column_name] = df[column_name].diff()
                    # Replace first NaN with a reasonable value (average of next few values)
                    if not df.empty:
                        df.iloc[0] = df.iloc[1:5].mean()
                    # Clean up any negative values
                    df[df[column_name] < 0] = 0
                
                all_data.append(df)
            else:
                print(f"Warning: No 'values' data found for {metric}")
        else:
            print(f"Error fetching {metric}: {response.status_code}")

    if all_data:
        final_df = pd.concat(all_data, axis=1).reset_index()
        
        # Calculate transaction fees in USD
        final_df['transaction_fees_usd'] = final_df['transaction_fees_btc'] * final_df['btc_market_price_usd']
        
        return final_df
    else:
        print("Error: No valid blockchain data retrieved.")
        return pd.DataFrame()

# Fetch all data
data = fetch_blockchain_data()
if data.empty:
    print("No blockchain data available. Exiting script.")
    exit()

# Ensure timestamp is in UTC
data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)

# Fetch energy prices
denmark_prices = fetch_denmark_prices(start_time_str, end_time_str)
texas_prices = fetch_texas_prices()
kazakhstan_prices = fetch_kazakhstan_prices()
china_prices = fetch_china_prices()

# Merge energy prices if available
if not denmark_prices.empty:
    data = pd.merge_asof(data, denmark_prices, on='timestamp', direction='nearest')
    print("Successfully merged Denmark prices")
else:
    print("Warning: No Denmark price data available")

if not texas_prices.empty:
    data = pd.merge_asof(data, texas_prices, on='timestamp', direction='nearest')
    print("Successfully merged Texas prices")
else:
    print("Warning: No Texas price data available")

if not kazakhstan_prices.empty:
    data = pd.merge_asof(data, kazakhstan_prices, on='timestamp', direction='nearest')
    print("Successfully merged Kazakhstan prices")
else:
    print("Warning: No Kazakhstan price data available")

if not china_prices.empty:
    data = pd.merge_asof(data, china_prices, on='timestamp', direction='nearest')
    print("Successfully merged China prices")
else:
    print("Warning: No China price data available")

# Format timestamp for CSV
data['timestamp'] = data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Convert float formatting for Excel compatibility with 2 decimal places
for column in data.select_dtypes(include=['float64']).columns:
    data[column] = data[column].map(lambda x: f"{x:.2f}".replace('.', ','))

# Save to CSV
data.to_csv('bitcoin_metrics_and_energy.csv', 
            index=False,
            sep=';',
            encoding='utf-8-sig')

print("\nColumns in final dataset:")
print(data.columns.tolist())
print("\nData shape:", data.shape)
print("\nFirst few rows:")
print(data.head())
