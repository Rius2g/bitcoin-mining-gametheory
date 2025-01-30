import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from profit import *
import matplotlib as mpl

# parameters
font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 20}

mpl.rc('font',**font)
linewidth = 3

# Generate data for the plot
# tx_fee_reduction = np.linspace(0.01, 0.8, 100)  # TX fee reduction from 1% to 80%
tx_fee_reduction = np.asarray([0.01, 0.1, 0.25, 0.5])

# el_price_categories = [0.032, 0.162, 0.469]  # Different electricity price categories
el_price_categories = [0.032, 0.162]
el_prices_label = ['Qatar', 'US', 'Denmark']
# btc_price_categories = [10000, 25000, 100000]  # Different BTC price categories
btc_price_categories = [10000, 30000, 60000]

# Define color and line patterns
color_map = {el_price: f'C{i}' for i, el_price in enumerate(el_price_categories)}

markers = ['o', 's', '^']
line_patterns = ['-', '--', ':']

# Plotting lines for different categories of el_price and btc_price
plt.figure(figsize=(10, 6))
for i, el_price in enumerate(el_price_categories):
    for j, btc_price in enumerate(btc_price_categories):
        miner_profit = [profit(el_price, btc_price, m=sum_tx_fees(fpt=FEE_PER_TX*(1-fee))) for fee in tx_fee_reduction]
        plt.plot(tx_fee_reduction * 100, miner_profit, label=f'El Price: ${el_price}, BTC Price: ${btc_price}',
                 color=color_map[el_price], marker=markers[i], linestyle=line_patterns[j], linewidth=linewidth)

plt.xlim(0, 50)  # Set x-axis limits

# Create personalized legend
legend_handles = []
for i, el_price in enumerate(el_price_categories):
    legend_handles.append(plt.Line2D([], [], color=color_map[el_price], marker=markers[i], label=f'{el_prices_label[i]}'))
for j, btc_price in enumerate(btc_price_categories):
    legend_handles.append(plt.Line2D([], [], linestyle=line_patterns[j], color='black', label=f'${btc_price} x BTC'))

plt.xlabel('TX Fee Reduction (%)')
plt.ylabel('Miner Profit ($)')
plt.title('Miner Profit vs. Fee Reduction')
plt.grid(True)
plt.legend(handles=legend_handles)
plt.show()
