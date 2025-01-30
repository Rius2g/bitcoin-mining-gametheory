import pandas as pd
import numpy as np
from profit import *
import json
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import matplotlib as mpl

font = {'family': 'Times New Roman',
        'weight': 'bold',
        'size': 46}

mpl.rc('font', **font)


# epochs data:
jan2015 = 1420120386
jan2016 = 1451646017
jan2017 = 1483278786
jan2018 = 1514814786
jan2019 = 1546350786
jan2020 = 1577886786
jan2021 = 1609509186
jan2022 = 1641037137
# timestamp of change in block reward
br = [1353888000, 1464998400, 1589155200]  # 50 to 25 || 25 to 12.5 || 12.5 to 6.25


# miner class
class Miner:
    def __init__(self, efficiency, h_rate, start_date, divide_for_hash=1000000000000):
        self.efficiency = efficiency
        self.hr = h_rate
        self.start = start_date
        self.cph = self.cph(divide_for_hash)
        self.cpd = self.cpd()
        self.watt = self.calculate_watt()

    def cph(self, divide_for_hash):
        # consumption per hash
        return ((self.efficiency / (60 * 60)) / 1000) / divide_for_hash

    def cpd(self):
        # consumption per day
        return self.cph * 60 * 60 * 24 * self.hr

    def calculate_watt(self):
        return self.cpd / 24 * 1000


antminerS19 = Miner(efficiency=29.55, h_rate=110000000000000, start_date=1651404815)  # 2020
antminerS9 = Miner(efficiency=98.21, h_rate=13500000000000, start_date=1504265615)  # 2017
antminerS7 = Miner(efficiency=250, h_rate=4730000000000, start_date=1441112338)  # 2015


min_date = jan2016  # 1-Jan-16
max_date = jan2022  # 1-Jan-22

# individual hash rate
ihr = 110000000000000

# fetch market price
mrk_price_json = pd.read_json("json/market-price.json")
market_price = pd.DataFrame()
market_price['price'] = mrk_price_json.apply(lambda x: x['market-price']['y'], axis=1)
market_price['epoch'] = mrk_price_json.apply(lambda x: x['market-price']['x'], axis=1)

# fetch total hash rate
with open("json/hash-rate.json") as json_data:
    hash_pr_json = json.load(json_data)['hash-rate']
hash_rate = pd.DataFrame()
hash_rate['hash-rate'] = [x['y'] for x in hash_pr_json]  # in TH
hash_rate['hash-rate'] = hash_rate.apply(lambda x: x['hash-rate'] * (10 ** 12), axis=1)
hash_rate['epoch'] = [x['x'] for x in hash_pr_json]

# fetch transaction fees
with open("json/transaction-fees-usd.json") as json_data:
    tx_fee_json = json.load(json_data)['transaction-fees-usd']
tx_fees = pd.DataFrame()
tx_fees['fee'] = [x['y'] for x in tx_fee_json]
tx_fees['epoch'] = [x['x'] for x in tx_fee_json]

# fetch avg transaction fee per day
with open("json/fees-usd-per-transaction.json") as json_data:
    avg_tx_fee_json = json.load(json_data)['fees-usd-per-transaction']
avg_tx_fee = pd.DataFrame()
avg_tx_fee['avg-fee'] = [x['y'] for x in avg_tx_fee_json]
avg_tx_fee['epoch'] = [x['x'] for x in avg_tx_fee_json]

# fetch electricity price
el_prices = [0.1252, 0.1265, 0.1255, 0.1289, 0.1287, 0.1301, 0.1315, 0.1366, 0.162]  # 2014-2022

# merge data together
merged_dataframe = pd.merge_asof(hash_rate, market_price, on="epoch")
merged_dataframe = pd.merge_asof(merged_dataframe, tx_fees, on="epoch")
cost = pd.merge_asof(merged_dataframe, avg_tx_fee, on="epoch")
cost["fee"] = cost["fee"].replace(np.nan, 0)
cost["avg-fee"] = cost["avg-fee"].replace(np.nan, 0)
cost['epoch'] = cost.apply(lambda x: int(x['epoch']) / 1000, axis=1)
# add el prices
cost['el_price'] = cost.apply(lambda row: el_prices[0] if row['epoch'] <= jan2015 else
(el_prices[1] if row['epoch'] <= jan2016 else (el_prices[2] if row['epoch'] <= jan2017 else
                                               (el_prices[3] if row['epoch'] <= jan2018 else
                                                (el_prices[4] if row['epoch'] <= jan2019 else
                                                 (el_prices[5] if row['epoch'] <= jan2020 else
                                                  (el_prices[6] if row['epoch'] <= jan2021 else
                                                   (el_prices[7] if row['epoch'] <= jan2022 else
                                                    el_prices[8]))))))), axis=1)

# add different miners
cost['miner'] = cost.apply(lambda row: 'antminerS7' if row['epoch'] <= antminerS9.start else
('antminerS9' if row['epoch'] <= antminerS19.start else 'antminerS19'), axis=1)

# calculate costs
cost['cost'] = cost.apply(lambda row: costs(row['el_price'], cpd=antminerS7.cpd) if row['miner'] == 'antminerS7' else
(costs(row['el_price'], cpd=antminerS9.cpd) if row['miner'] == 'antminerS9' else costs(row['el_price'],
                                                                                       cpd=antminerS19.cpd)), axis=1)

# insert reward
cost['reward'] = cost.apply(lambda row: 50 * row['price'] if row['epoch'] <= br[0] else
(25 * row['price'] if row['epoch'] <= br[1] else (12.5 * row['price'] if row['epoch'] <= br[2]
                                                 else 6.25 * row['price'])), axis=1)

# calculate revenue
cost['revenue-0-fee'] = cost.apply((lambda row: revenue(row['price'],
                                                        mhr=miner_hash_ratio(btchr=row['hash-rate'], ihr=antminerS7.hr),
                                                        block_reward=row['reward'], reward_in_usd=True,
                                                        m=sum_tx_fees(fpt=0)) if row['miner'] == 'antminerS7'
                                    else(revenue(row['price'],
                                                        mhr=miner_hash_ratio(btchr=row['hash-rate'], ihr=antminerS9.hr),
                                                        block_reward=row['reward'], reward_in_usd=True,
                                                        m=sum_tx_fees(fpt=0)) if row['miner'] == 'antminerS9'
                                         else revenue(row['price'],
                                                        mhr=miner_hash_ratio(btchr=row['hash-rate'], ihr=antminerS19.hr),
                                                        block_reward=row['reward'], reward_in_usd=True,
                                                        m=sum_tx_fees(fpt=0)))), axis=1)
cost['revenue'] = cost.apply((lambda row: revenue(row['price'],
                                                        mhr=miner_hash_ratio(btchr=row['hash-rate'], ihr=antminerS7.hr),
                                                        block_reward=row['reward'], reward_in_usd=True,
                                                        m=sum_tx_fees(fpt=row['avg-fee'])) if row['miner'] == 'antminerS7'
                                    else(revenue(row['price'],
                                                        mhr=miner_hash_ratio(btchr=row['hash-rate'], ihr=antminerS9.hr),
                                                        block_reward=row['reward'], reward_in_usd=True,
                                                        m=sum_tx_fees(fpt=row['avg-fee'])) if row['miner'] == 'antminerS9'
                                         else revenue(row['price'],
                                                        mhr=miner_hash_ratio(btchr=row['hash-rate'], ihr=antminerS19.hr),
                                                        block_reward=row['reward'], reward_in_usd=True,
                                                        m=sum_tx_fees(fpt=row['avg-fee'])))), axis=1)
# cost['revenue'] = cost.apply((lambda row: revenue(row['price'], mhr=miner_hash_ratio(btchr=row['hash-rate'], ihr=ihr),
#                                                   block_reward=row['reward'], reward_in_usd=True,
#                                                   m=sum_tx_fees(fpt=row['avg-fee']))), axis=1)

# calculate profits
cost['profit-0-fee'] = cost.apply(lambda row: row['revenue-0-fee'] - row['cost'], axis=1)
cost['profit'] = cost.apply(lambda row: row['revenue'] - row['cost'], axis=1)

# to plot
cost = cost[cost['epoch'] >= min_date]
plot_df = cost[cost['epoch'] <= max_date]

# plot profit over txs-fees
color_ax1 = 'royalblue'
color_ax2 = 'orange'
import matplotlib.dates as mdate

x = np.asarray(plot_df['epoch'])
# Convert to the correct format for matplotlib.
# mdate.epoch2num converts epoch timestamps to the right format for matplotlib
x = mdate.epoch2num(x)
y1 = np.asarray(plot_df['profit-0-fee'])
y2 = np.asarray(plot_df['avg-fee'])

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

lns1 = ax1.plot_date(x, y1, '-', color=color_ax1, label='Profit 0 fee')
lns2 = ax2.plot_date(x, y2, '-', color=color_ax2, label='Avg fee/day')
lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=9)

# Choose your xtick format string
date_fmt = '%b-%y'
# Use a DateFormatter to set the data to the correct format.
date_formatter = mdate.DateFormatter(date_fmt)
ax1.xaxis.set_major_formatter(date_formatter)
ax2.xaxis.set_major_formatter(date_formatter)
# Sets the tick labels diagonal so they fit easier.
fig.autofmt_xdate()

# ax1.set_xlabel('Date')
ax1.set_ylabel('USD')
# ax2.set_ylabel('Fee ($)', color=color_ax2)

ax1.fill_between(x, y1, step="pre", alpha=0.3, color=color_ax1)
ax2.fill_between(x, y2, step="pre", alpha=0.3, color=color_ax2)
# ax1.set_ylim(bottom=0)
# ax2.set_ylim(bottom=0)
ax1.set_ylim(-5, 30)
ax2.set_ylim(-5, 30)
ax2.tick_params(left=False, labelleft=False, top=False, labeltop=False,
                   right=False, labelright=False, bottom=False, labelbottom=False)
ax1.grid()

plt.show()
