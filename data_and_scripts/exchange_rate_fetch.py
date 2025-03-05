import requests
import pandas as pd
from datetime import datetime, timedelta

# Define date range
start_date = "2023-02-08"
end_date = "2025-02-08"

# Fetch exchange rate data from ECB (Euro to USD)
url = f"https://api.frankfurter.app/{start_date}..{end_date}?from=EUR&to=USD"
response = requests.get(url)
data = response.json()

# Convert to DataFrame
df = pd.DataFrame.from_dict(data["rates"], orient="index")
df.reset_index(inplace=True)
df.columns = ["Date", "Exchange Rate"]

# Convert Date column to datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Generate a full date range (including weekends)
full_date_range = pd.DataFrame({"Date": pd.date_range(start=start_date, end=end_date)})

# Merge with fetched data & interpolate missing weekend values
df = full_date_range.merge(df, on="Date", how="left")
df["Exchange Rate"] = df["Exchange Rate"].interpolate()

# Save to Excel
df.to_excel("exchange_rates_daily.xlsx", index=False)

print("Daily exchange rates saved successfully!")

