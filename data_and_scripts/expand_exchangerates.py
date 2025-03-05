import pandas as pd

# Read the daily exchange rates Excel file
df = pd.read_excel('exchange_rates_daily.xlsx')

# Create a list of time intervals (as strings or actual times)
# Example: ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']
time_intervals = ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']

# Repeat each row 8 times
df_repeated = df.loc[df.index.repeat(8)].copy()
df_repeated.reset_index(drop=True, inplace=True)

# Now add a new column for the 3-hour interval
# The idea is to assign each repeated block of 8 rows a different time from our list.
df_repeated['Time'] = time_intervals * (len(df_repeated) // 8)

# Optionally, if you want to combine date with time, assuming there's a date column:
if 'Date' in df_repeated.columns:
    df_repeated['Datetime'] = pd.to_datetime(df_repeated['Date'].astype(str) + ' ' + df_repeated['Time'])

# Save the new DataFrame to a new Excel file if desired:
df_repeated.to_excel('exchange_rates_intra_day.xlsx', index=False)

print("Data has been transformed to 3-hour intervals.")

