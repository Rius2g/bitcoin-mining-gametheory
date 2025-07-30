import pandas as pd

# Rig specs data
model = [
    "S19 Pro", "S19j Pro", "T19 Hydro", "S19 Hydro", "S19 XP Hyd",
    "S19j Pro+", "S19 Pro Hyd", "S19K Pro", "T21", "S21", "S21 Hyd"
]
capacity_THs = [
    110, 100, 145, 158, 255,
    122, 177, 136, 190, 200, 335
]
efficiency = [
    30, 31, 38, 34, 21,
    28, 29, 24, 19, 17, 16
]
df_rig = pd.DataFrame({
    'model': model,
    'capacity_THs': capacity_THs,
    'efficiency': efficiency
})
df_rig.to_excel('rig_specs.xlsx', index=False)

# Country/cost data
countries = [
    "Italy", "United Kingdom", "Australia", "Japan", "United States", "Brazil", "South Africa", "India", "Russia", "Venezuela", "Qatar", "Iran"
]
electricity_cost_usd_per_kwh = [
    0.31, 0.30, 0.29, 0.27, 0.15, 0.17, 0.22, 0.08, 0.06, 0.05, 0.03, 0.002
]
df_cost = pd.DataFrame({
    'country': countries,
    'cost': electricity_cost_usd_per_kwh
})
df_cost.to_excel('cost_rates.xlsx', index=False)

print("Wrote rig_specs.xlsx and cost_rates.xlsx")

