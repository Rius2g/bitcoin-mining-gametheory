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
        'size'   : 48}

mpl.rc('font',**font)
linewidth = 3

title_qatar = "QATAR"
title_us = "US"
title_dk = "DENMARK"
title_list = [title_qatar, title_us, title_dk]

total_hashrate_list = np.linspace(50000000000000000000, 800000000000000000000, 1000)  # total H
h_list = np.linspace(10000000000000, 750000000000000, 1000)  # individual Hash rate h

# electricity  prices
qatar = 0.032
us = 0.162
dk = 0.469
el_prices = [qatar, us, dk]

# bitcoin price
btc_price = 15000

# margin of profit
margin = 10

# fee list
modified_fee_list = [1, 70]
fees_list = modified_fee_list

# make x and y
x_val, y_val = np.meshgrid(total_hashrate_list, h_list)

# contour
nrows = len(fees_list) * len(el_prices)

new_el_price_list = []
new_fee_list = []
new_title_list = []
i = 0
for el in el_prices:
    for f in fees_list:
        new_el_price_list.append(el)
        new_fee_list.append(f)
        new_title_list.append(title_list[i])
    i += 1

fig, axes = plt.subplots(nrows=3, ncols=2)
i = 0
min_profit = -50
max_profit = 100
levels = np.linspace(min_profit, max_profit, 30)
for ax in axes.flat:
    z_val = profit(new_el_price_list[i], btc_price, mhr=miner_hash_ratio(btchr=x_val, ihr=y_val),
                   cpd=consumption_per_day(ihr=y_val),
                   m=sum_tx_fees(2000, new_fee_list[i]))
    max_val = max([max(z) for z in z_val])
    min_val = min([min(z) for z in z_val])
    print(max_val)
    print(min_val)
    norm = mcolors.TwoSlopeNorm(vmin=-50, vmax=100, vcenter=0)
    im = ax.contourf(x_val, y_val, z_val, 500, cmap=cm.RdBu, norm=norm, levels=levels, extend='both')
    # ax.set_xticks([5e+19, 2e+20, 4e+20, 6e+20, 8e+20])
    ax.set_xticks([5e+19, 4e+20, 6e+20, 8e+20])
    ax.set_xticklabels([])
    ax.minorticks_off()

    # text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
    textstr = 'fee $' + str(new_fee_list[i])
    ax.text(0.98, 0.94, textstr, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    i += 1

fig.colorbar(im, ax=axes.ravel().tolist(), ticks=[-50, -25, 0, 25, 50, 100])

# color btc hashrate and miner hashrate
btchr = 2.6e+20
ihr = 1.1e+14

# plt.sca(axes[-1])
# plt.xticks([5e+19, 2e+20, 4e+20, 6e+20, 8e+20])

loc_x = 0
loc_y = 0
i = 0
for ax in axes.flat:
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    # append labels of total hash rate and individual hash rate
    xticks = np.append(xticks, btchr)
    yticks = np.append(yticks, ihr)
    xticks = np.sort(xticks)
    yticks = np.sort(yticks)
    # locate them
    loc_x = np.where(xticks == btchr)[0][0]
    loc_y = np.where(yticks == ihr)[0][0]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    # set gridlines
    ax.grid()
    x_gridline = ax.get_xgridlines()
    b = x_gridline[loc_x]
    b.set_color('red')
    y_gridline = ax.get_ygridlines()
    b = y_gridline[loc_y]
    b.set_color('red')
    if i < nrows-len(fees_list):
        # last row
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    else:
        # ax.set_xticklabels([50, 200, 260, 400, 600, 800])
        ax.set_xticklabels([50, 260, 400, 600, 800])
        ax.set_yticklabels([10, 110, 250, 500, 750])
        # color labels
        ax.get_yticklabels()[loc_y].set_color('red')
        ax.get_xticklabels()[loc_x].set_color('red')
    ax.set_ylim(ymin=min(h_list))
    ax.set_xlim(xmin=min(total_hashrate_list))
    ax.set_ylim(ymax=max(h_list))
    ax.set_xlim(xmax=max(total_hashrate_list))
    # set axes labels
    if i % len(fees_list) == 0:
        # ax.set_title(new_title_list[i], fontweight='bold', loc='left')
        ax.text(y=760e12, x=0.6e+20, s=new_title_list[i], fontweight='bold', ha='left')
        if new_el_price_list[i] == us:
            ax.set_ylabel(r'$h$ (TH/s)')
    if i >= nrows-len(fees_list):
        ax.set_xlabel(r'$H$ (EH/s)')
    i += 1
plt.show()

