import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from profit import *
from plot import *
import pprint
from labellines import labelLines
import matplotlib.colors as mcolors

# parameters
font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 34}

mpl.rc('font',**font)
linewidth = 3

price_list = np.linspace(100, 100000, 1000)  # BTC price list x-axis
fee_list = np.linspace(0, 150, 1000)  # fees y-axis

# electricity  prices
qatar = 0.032
us = 0.162
dk = 0.469
el_price = us
el_prices = [qatar, us, dk]

bhr = 260000000000000000000  # 30 day average
ihr = 110000000000000  # Antminer S19 Pro

# make x and y
x_val, y_val = np.meshgrid(price_list, fee_list)

i = 0
min_profit = -10
max_profit = 20
levels = np.linspace(min_profit, max_profit, 30)
z_val = profit(el_price=el_price, btc_price=x_val, m=sum_tx_fees(2000, y_val),
               cpd=consumption_per_day(ihr=ihr), mhr=miner_hash_ratio(btchr=bhr, ihr=ihr))
max_val = max([max(z) for z in z_val])
min_val = min([min(z) for z in z_val])
print(max_val)
print(min_val)
norm = mcolors.TwoSlopeNorm(vmin=min_profit, vmax=max_profit, vcenter=0)
im = plt.contourf(x_val, y_val, z_val, 500, cmap=cm.RdBu, norm=norm, levels=levels, extend='both')
# plt.set_xticks([5e+19, 4e+20, 6e+20, 8e+20])
# plt.set_xticklabels([])
# plt.minorticks_off()
plt.colorbar(im, ticks=[-10, -5, 0, 5, 10, 20])

plt.ylabel(r'fee ($)')
plt.xlabel(r'BTC price ($)')
plt.gcf().subplots_adjust(bottom=0.15)
plt.grid()
plt.show()

