import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from profit import *

# calculate table for electricity price and BTC price
el_prices = [0.032, 0.162, 0.469]
btc_prices = [1000, 10000, 15000, 30000, 60000, 100000]
for el in el_prices:
    for btc_p in btc_prices:
        pr = profit(el, btc_p)
        print("el price: $" + str(el) + ", BTC price: " + str(btc_p) + ", Profit: $" + str(pr))


df = pd.read_excel('costs.xlsx')
btc_price = 20000
el_price = 0.03
x_1 = np.linspace(50000000000000000000, 400000000000000000000, 1000)  # total H
x_2 = np.linspace(30000000000000, 1000000000000000, 1000)  # individual Hash rate h

y_1 = []
y_2 = []
for x1, x2 in zip(x_1, x_2):
    mhr1 = miner_hash_ratio(btchr=x1)
    mhr2 = miner_hash_ratio(ihr=x2)
    cpd = consumption_per_day(ihr=x2)
    y_1.append(profit(el_price, btc_price, mhr=mhr1))
    y_2.append(profit(el_price, btc_price, cpd=cpd, mhr=mhr2))

#fig = plt.figure()
fig, ax1 = plt.subplots()
ax2 = ax1.twiny()
ax1.plot(x_1, y_1, 'g-')
ax2.plot(x_2, y_2, 'b-')

ax1.set_xlabel('H (h/s)', color='g')
ax1.set_ylabel('Profit ($)')
ax2.set_xlabel('h (h/s)', color='b')

#x_val1 = df['Electricity Price ($ x kWh)']
#y_val1 = df['Day Cost ($)']
#x_val2 = df['BTC Price ($)']
#y_val2 = df['Daily Revenue ($)']

#ax = fig.add_subplot(111, label="Increasing H")
#ax2 = fig.add_subplot(111, label="Increasing h", frame_on=False)

#ax.plot(x_1, y_1, color="C0")
#ax.invert_xaxis()
#ax.set_xlabel("H ratio", color="C0")
#ax.set_ylabel("Profit", color="C0")
#ax.tick_params(axis='x', colors="C0")
#ax.tick_params(axis='y', colors="C0")

#ax2.plot(x_2, y_2, color="C1")
#ax2.xaxis.tick_top()
#ax2.yaxis.tick_right()
#ax2.set_xlabel("h power", color="C1")
#ax2.set_ylabel("Profit", color="C1")
#ax2.xaxis.set_label_position('top')
#ax2.yaxis.set_label_position('right')
#ax2.tick_params(axis='x', colors="C1")
#ax2.tick_params(axis='y', colors="C1")
plt.show()
