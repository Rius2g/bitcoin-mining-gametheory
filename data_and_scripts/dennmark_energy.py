import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
import time

def fetch_historical_eur_usd_rates(start_date, end_date):
    """
    Fetch daily historical EUR to USD exchange rates using Frankfurter API.
    """
    url = f"https://api.frankfurter.app/{start_date}..{end_date}?from=EUR&to=USD"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        rates = data['rates']
        df_rates = pd.DataFrame([(date, rate['USD']) for date, rate in rates.items()],
                                columns=['date', 'eur_usd_rate'])
        df_rates['date'] = pd.to_datetime(df_rates['date'])
        return df_rates
    except requests.exceptions.RequestException as e:
        print(f"Error fetching exchange rates: {e}")
        # Return a DataFrame with a default rate as fallback
        return pd.DataFrame({'date': [start_date], 'eur_usd_rate': [1.08]})

def fetch_denmark_prices(start_time_str, end_time_str):
    """
    Fetch Denmark energy prices with simplified approach.
    """
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
        
        # Convert timestamp and get date for FX rate matching
        df['timestamp'] = pd.to_datetime(df['HourUTC'], utc=True)
        df['date'] = df['timestamp'].dt.date
        
        # Get EUR/USD rates for the date range
        start_date = df['date'].min().strftime('%Y-%m-%d')
        end_date = df['date'].max().strftime('%Y-%m-%d')
        eur_usd_rates = fetch_historical_eur_usd_rates(start_date, end_date)
        
        # Convert date column to datetime for merging
        df['date'] = pd.to_datetime(df['date'])
        
        # Merge with exchange rates
        df = df.merge(eur_usd_rates, on='date', how='left')
        
        # Fill any missing rates with 1.08 as fallback
        df['eur_usd_rate'] = df['eur_usd_rate'].fillna(1.08)
        
        # Round timestamps to 3-hour intervals
        df['timestamp'] = df['timestamp'].dt.floor('3h')
        
        # Convert price from EUR/MWh to USD/kWh using dynamic rate
        df['denmark_price_usd_kwh'] = df['SpotPriceEUR'] / 1000 * df['eur_usd_rate']
        
        # Remove duplicates and keep only needed columns
        df = df[['timestamp', 'denmark_price_usd_kwh']].drop_duplicates(subset=['timestamp'])
        
        return df
    else:
        print("Error fetching Denmark prices:", response.status_code, response.text)
        return pd.DataFrame()

if __name__ == "__main__":
    # Calculate date range (2 years back from today)
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=730)
    
    start_time_str = start_time.strftime('%Y-%m-%dT%H:%M')
    end_time_str = end_time.strftime('%Y-%m-%dT%H:%M')
    
    print(f"Fetching data from {start_time_str} to {end_time_str}")
    
    denmark_prices = fetch_denmark_prices(start_time_str, end_time_str)
    
    if not denmark_prices.empty:
        # Basic validation
        expected_rows = (end_time - start_time) / timedelta(hours=3)
        actual_rows = len(denmark_prices)
        coverage = actual_rows / expected_rows
        
        print(f"\nData coverage analysis:")
        print(f"Expected rows: {expected_rows:.0f}")
        print(f"Actual rows: {actual_rows}")
        print(f"Coverage: {coverage:.2%}")
        
        if coverage >= 0.8:  # 80% or better coverage
            output_file = 'denmark_prices_dynamic_fx_full_range.csv'
            denmark_prices.to_csv(output_file, index=False)
            print(f"\nSuccessfully saved {actual_rows} rows to {output_file}")
            print(f"Date range: {denmark_prices['timestamp'].min()} to {denmark_prices['timestamp'].max()}")
        else:
            print(f"\nWarning: Low data coverage ({coverage:.2%})")
            print("Saving data anyway for inspection...")
            denmark_prices.to_csv('denmark_prices_partial.csv', index=False)
    else:
        print("\nNo data to save")
