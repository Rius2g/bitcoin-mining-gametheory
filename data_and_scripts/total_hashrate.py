import requests
import pandas as pd
import os
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

# Calculate timestamps (2 years back from today)
end_time = datetime.now(timezone.utc)
start_time = end_time - timedelta(days=2 * 365)

# Convert timestamps to ISO 8601 format (without seconds) for Denmark API
start_time_str = start_time.strftime("%Y-%m-%dT%H:%M")
end_time_str = end_time.strftime("%Y-%m-%dT%H:%M")

# Function to fetch Denmark electricity prices
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

# Function to generate static Texas electricity prices
def fetch_texas_prices():
    static_texas_price = 0.114  # USD/kWh
    timestamps = pd.date_range(start=start_time, end=end_time, freq='3h', tz="UTC")
    df = pd.DataFrame({'timestamp': timestamps, 'texas_price_usd_kwh': static_texas_price})
    return df

# Function to fetch Kazakhstan electricity prices
def fetch_kazakhstan_prices():
    avg_price_per_kwh_usd = 0.035  # Estimated price
    timestamps = pd.date_range(start=start_time, end=end_time, freq='3h', tz="UTC")
    df = pd.DataFrame({'timestamp': timestamps, 'kazakhstan_price_usd_kwh': avg_price_per_kwh_usd})
    return df

# Function to fetch blockchain data
def fetch_blockchain_data():
    metrics = {
        'hash-rate': 'hash_rate',
        'market-price': 'btc_market_price_usd'
    }
    
    all_data = []
    
    for metric, column_name in metrics.items():
        url = f'https://api.blockchain.info/charts/{metric}?timespan=2years&format=json'
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'values' in data:
                df = pd.DataFrame(data['values'])
                df.columns = ['timestamp', column_name]
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
                df = df.set_index('timestamp')
                all_data.append(df)
            else:
                print(f"Warning: No 'values' data found for {metric}")
        else:
            print(f"Error fetching {metric}: {response.status_code}")

    if all_data:
        final_df = pd.concat(all_data, axis=1).reset_index()
        return final_df
    else:
        print("Error: No valid blockchain data retrieved.")
        return pd.DataFrame()

# Fetch and process blockchain data
data = fetch_blockchain_data()

# Prevent merge if blockchain data is empty
if data.empty:
    print("No blockchain data available. Exiting script.")
    exit()

# Resample blockchain data to exactly 3-hour intervals
data = data.set_index('timestamp')
data = data.resample('3h').mean()
data = data.interpolate(method='linear')
data = data.reset_index()

# Fetch energy price data
denmark_prices = fetch_denmark_prices(start_time_str, end_time_str)
texas_prices = fetch_texas_prices()
kazakhstan_prices = fetch_kazakhstan_prices()

# Ensure all datasets have timestamps in UTC
data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)

data = pd.merge_asof(data, denmark_prices, on='timestamp', direction='nearest')
data = pd.merge_asof(data, texas_prices, on='timestamp', direction='nearest')
data = pd.merge_asof(data, kazakhstan_prices, on='timestamp', direction='nearest')

# Debugging output after merge
print("\nMerged Data Sample (After Fixing Alignment):")
print(data.head(20))

# Format timestamp for CSV
data['timestamp'] = data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Save to CSV
data.to_csv('bitcoin_energy_prices.csv', 
            index=False,
            sep=';',
            decimal='.',
            encoding='utf-8-sig',
            float_format='%.2f')

print("Data shape:", data.shape)
print("\nFirst few rows:")
print(data.head())
