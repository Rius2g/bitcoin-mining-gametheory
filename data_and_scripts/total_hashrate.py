import pandas as pd
import requests
from datetime import datetime, timedelta
import time

# Calculate timestamps
end_time = int(time.time())
start_time = end_time - (2 * 365 * 24 * 60 * 60)  # 2 years in seconds

#should also get the energy prices for Texas, Dennmark and Khazakstan
#Lastly if possible get the fees between blocks, this way we can calculate and map the profitability of mining over the last 2 years
def fetch_blockchain_data(start_time, end_time, interval=10800):
    metrics = {
        'hash-rate': 'hash_rate',
        'market-price': 'market_price_usd'
    }
    
    all_data = []
    
    for metric, column_name in metrics.items():
        url = f'https://api.blockchain.info/charts/{metric}'
        params = {
            'start': start_time,
            'end': end_time,
            'timespan': '2years',
            'sampledIn': interval,
            'format': 'json'
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data['values'])
            df.columns = ['timestamp', column_name]
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.set_index('timestamp')
            all_data.append(df)
        else:
            print(f"Error fetching {metric}: {response.status_code}")
    
    final_df = pd.concat(all_data, axis=1)
    final_df = final_df.reset_index()
    return final_df

# Fetch and process data
data = fetch_blockchain_data(start_time, end_time)

# Resample to exactly 3-hour intervals and interpolate missing values
data = data.set_index('timestamp')
data = data.resample('3H').mean()
data = data.interpolate(method='linear')
data = data.reset_index()

# Format timestamp to be more Excel-friendly
data['timestamp'] = data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Ensure columns are in desired order
data = data[['timestamp', 'hash_rate', 'market_price_usd']]

# Save to CSV with Excel-friendly formatting
data.to_csv('bitcoin_metrics_3h.csv', 
            index=False,
            sep=';',          # Change to semicolon separator for Excel
            decimal='.',      # Use period for decimal point
            encoding='utf-8-sig',  # Add BOM for Excel
            float_format='%.2f')

print("Data shape:", data.shape)
print("\nFirst few rows:")
print(data.head())
