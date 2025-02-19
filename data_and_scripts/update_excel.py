import pandas as pd
from datetime import datetime

# Read both files
denmark_prices = pd.read_csv('denmark_prices_dynamic_fx_full_range.csv')
excel_data = pd.read_excel('bitcoin_metrics_and_energy_updated.xlsx')

# Convert timestamps to consistent format
denmark_prices['timestamp'] = pd.to_datetime(denmark_prices['timestamp']).dt.strftime('%Y-%m-%d %H:00:00')
excel_data['timestamp'] = pd.to_datetime(excel_data['timestamp']).dt.strftime('%Y-%m-%d %H:00:00')

# Create a dictionary of denmark prices
denmark_price_dict = dict(zip(denmark_prices['timestamp'], denmark_prices['denmark_price_usd_kwh']))

# Update the denmark prices in excel data
excel_data['denmark price usd kwh'] = excel_data['timestamp'].map(denmark_price_dict).fillna(excel_data['denmark price usd kwh'])

# Save the updated Excel file
excel_data.to_excel('bitcoin_metrics_and_energy_updated_final.xlsx', index=False)

# Print summary
total_updates = sum(excel_data['timestamp'].isin(denmark_prices['timestamp']))
print(f"\nUpdate Summary:")
print(f"Total rows in Excel: {len(excel_data)}")
print(f"Total Denmark prices available: {len(denmark_prices)}")
print(f"Rows updated with new prices: {total_updates}")
print("New file saved as: bitcoin_metrics_and_energy_updated_final.xlsx")
