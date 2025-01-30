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

df = pd.read_excel('costs.xlsx')

# times new roman
# csfont = {'fontname':'Times New Roman'}
font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 66}

mpl.rc('font',**font)


total_hashrate_list = np.linspace(50000000000000000000, 800000000000000000000, 1000)  # total H
h_list = np.linspace(10000000000000, 750000000000000, 1000)  # individual Hash rate h

# electricity  prices
qatar = 0.032
us = 0.162
dk = 0.469

el_price = us
btc_price = 15000
fee_per_tx = 0

if el_price == qatar:
    title_label = "QATAR"
elif el_price == us:
    title_label = "US"
else:
    title_label = "DENMARK"

# fees_list = np.around(np.logspace(0, 2, 5), decimals=1)
fees_list_qatar = [0, 50, 100]
fees_list_us = [0, 10, 50, 100, 150]
fees_list_dk = [50, 100, 300, 400]
standard_fee_list = [1, 50, 100]
modified_fee_list = [1, 70]
fees_list = modified_fee_list
print(fees_list)

# make color list
color_list = ['#A9A9A9', '#888888', '#606060', '#383838', '#000000']  # 20 colors in grayscale
# color_list = ['#A9A9A9', '#606060', '#000000']  # Qatar color scale
#x_val = df['Electricity Price ($ x kWh)']
#y_val = df['BTC Price ($)']
x_val, y_val = np.meshgrid(total_hashrate_list, h_list)
z_val = profit(el_price, btc_price, mhr=miner_hash_ratio(btchr=x_val, ihr=y_val), cpd=consumption_per_day(ihr=y_val),
               m=sum_tx_fees(2000, fee_per_tx))

# plotting different fee impact on mining
# fig = plt.figure()
# adjustFigAspect(fig, aspect=4)
# ax = fig.add_subplot(111)
linewidth = 3

# for fill between
x_arr = []
y_arr = []
margin = 10

# contour
nrows=len(fees_list)
fig, axes = plt.subplots(nrows=nrows, ncols=1)
i = 0
min_profit = -50
max_profit = 100
levels = np.linspace(min_profit, max_profit, 30)
for ax in axes.flat:
    z_val = profit(el_price, btc_price, mhr=miner_hash_ratio(btchr=x_val, ihr=y_val),
                   cpd=consumption_per_day(ihr=y_val),
                   m=sum_tx_fees(2000, fees_list[i]))
    max_val = max([max(z) for z in z_val])
    min_val = min([min(z) for z in z_val])
    print(max_val)
    print(min_val)
    norm = mcolors.TwoSlopeNorm(vmin=-50, vmax=100, vcenter=0)
    im = ax.contourf(x_val, y_val, z_val, 500, cmap=cm.RdBu, norm=norm, levels=levels, extend='both')
    ax.set_xticks([5e+19, 2e+20, 4e+20, 6e+20, 8e+20])
    ax.set_xticklabels([])
    ax.minorticks_off()
    if i == 1:
        # ax.set_ylabel('h (h/s)')
        pass
    # ax.label_outer()
    # ax.set_title("fee per tx : $" + str(fees_list[i]), fontsize=40)

    # text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
    textstr = 'fee $' + str(fees_list[i])
    ax.text(0.98, 0.90, textstr, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    i += 1
if el_price==us:
    fig.colorbar(im, ax=axes.ravel().tolist(), ticks=[-50, -25, 0, 25, 50, 100])
# fig.colorbar(plot, shrink=1, aspect=5)

# i = 0
# for fee in fees_list:
#         z_val = profit(el_price, btc_price, mhr=miner_hash_ratio(btchr=x_val, ihr=y_val),
#                        cpd=consumption_per_day(ihr=y_val),
#                        m=sum_tx_fees(2000, fee))
#         plot = ax.contourf(x_val, y_val, z_val, 1000, colors='w')  # cmap=cm.RdBu)
#         positive = find_nearest(plot.levels, margin)
#         negative = find_nearest(plot.levels, -margin)
#         cs = ax.contour(plot, linewidths=linewidth, levels=np.asarray([positive]), colors='black',
#                         origin='lower')
#
#         # fill between
#         retx, rety = get_xy_coordinates(cs)
#         x_arr.append(retx)
#         y_arr.append(rety)
#
#         # contour labels
#         ax.clabel(cs, cs.levels, inline=True, fmt=fmt_personalized(cs.levels, np.asarray([str(fee)])),
#                    manual=manual_labeling(ax, cs))
#         i += 1

# plot fill between
# i = 0
# for fee in fees_list:
#     # calculated formula from profit
#     a = ((6.25 * btc_price) + (fee * 2000)) * 144
#     b = 7.09e-13 * el_price
#     h = margin * (1 / ((a / total_hashrate_list) - b))
#     h = replace_negative(h, max(h_list))
#     ax.plot(total_hashrate_list, h, color=color_list[i], label=fees_list[i], linewidth=linewidth)
#     if i == 0:
#         # ax.fill_between(total_hashrate_list, h, max(h_list), color=color_list[i], alpha=.4)
#         pass
#     else:
#         a = ((6.25 * btc_price) + (fees_list[i-1] * 2000)) * 144
#         b = 7.09e-13 * el_price
#         h_prev = margin * (1 / ((a / total_hashrate_list) - b))
#         h_prev = replace_negative(h_prev, max(h_list))
#         # ax.fill_between(total_hashrate_list, h, h_prev, color=color_list[i], alpha=.4)
#     i += 1
# ax.legend(loc=0, title="Fee per tx ($)")

# ax.set_xlabel('H (h/s)')
# ax.set_ylabel('h (h/s)')

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
    if i != nrows-1:
        # last row
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    else:
        ax.set_xticklabels([50, 200, 260, 400, 600, 800])
        ax.set_yticklabels([10, 110, 250, 500, 750])
        # color labels
        ax.get_yticklabels()[loc_y].set_color('red')
        ax.get_xticklabels()[loc_x].set_color('red')
    ax.set_ylim(ymin=min(h_list))
    ax.set_xlim(xmin=min(total_hashrate_list))
    ax.set_ylim(ymax=max(h_list))
    ax.set_xlim(xmax=max(total_hashrate_list))
    # set axes labels
    if i == 0:
        ax.set_ylabel(r'$h$ (TH/s)')
    if i == nrows-1 and el_price==dk:
        ax.set_xlabel(r'$H$ (EH/s)')
    i += 1

    fig.suptitle(title_label, fontweight='bold', y=0.95, x=0.1, ha='left')
if el_price == dk:
    plt.gcf().subplots_adjust(bottom=0.15)
plt.show()


# # make labels red
# plt.gca().get_xticklabels()[loc_x].set_color('red')
# plt.gca().get_yticklabels()[loc_y].set_color('red')

# # set limits
# for ax in axes.flat:
#     ax.grid()
#     x_gridline = ax.get_xgridlines()
#     b = x_gridline[loc_x]
#     b.set_color('red')
#     y_gridline = ax.get_ygridlines()
#     b = y_gridline[loc_y]
#     b.set_color('red')
#     ax.get_yticklabels()[loc_y].set_color('red')
#     ax.get_xticklabels()[loc_x].set_color('red')
#     ax.set_ylim(ymin=min(h_list))
#     ax.set_xlim(xmin=min(total_hashrate_list))
#     ax.set_ylim(ymax=max(h_list))
#     ax.set_xlim(xmax=max(total_hashrate_list))


# working labels

# plt.sca(axes[-1])
# plt.xticks([5e+19, 2e+20, 4e+20, 6e+20, 8e+20])
#
# xticks = ax.get_xticks()
# yticks = ax.get_yticks()
#
# xticks = np.append(xticks, btchr)
# yticks = np.append(yticks, ihr)
#
# xticks = np.sort(xticks)
# yticks = np.sort(yticks)
#
# loc_x = np.where(xticks==btchr)[0][0]
# loc_y = np.where(yticks==ihr)[0][0]
#
# for ax in axes.flat:
#     ax.set_xticks(xticks)
#     ax.set_yticks(yticks)
#
# plt.gca().get_xticklabels()[loc_x].set_color('red')
# plt.gca().get_yticklabels()[loc_y].set_color('red')
#
# for ax in axes.flat:
#     ax.set_ylim(ymin=min(h_list))
#     ax.set_xlim(xmin=min(total_hashrate_list))
#     ax.set_ylim(ymax=max(h_list))
#     ax.set_xlim(xmax=max(total_hashrate_list))
#
# # labelLines(plt.gca().get_lines(), zorder=2.5)
#
# for ax in axes.flat:
#     ax.grid()
#     x_gridline = ax.get_xgridlines()
#     b = x_gridline[loc_x]
#     b.set_color('red')
#     y_gridline = ax.get_ygridlines()
#     b = y_gridline[loc_y]
#     b.set_color('red')
#     ax.get_yticklabels()[loc_y].set_color('red')
#     ax.get_xticklabels()[loc_x].set_color('red')
#
# fig.suptitle("Qatar")
# plt.show()

# working label until HERE

# 3d surface
#fig = plt.figure()
#ax = Axes3D(fig)
#surf = ax.plot_surface(x_val, y_val, z_val, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.3, aspect=5)
#plt.title('Profit')
#plt.show()

