import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BITCOIN_HASH_RATE = 260000000000000000000  # 30 day average
INDIVIDUAL_HASH_RATE = 110000000000000  # Antminer S19 Pro
MINER_POWER_EFFICIENCY = 29.55
BLOCK_REWARD = 6.25/2
TXS_PER_BLOCK = 2500
FEE_PER_TX = 7.5 - (7.5*0.1)


def sum_tx_fees(tpb=TXS_PER_BLOCK, fpt=FEE_PER_TX):
    return tpb * fpt


def miner_hash_ratio(btchr=BITCOIN_HASH_RATE, ihr=INDIVIDUAL_HASH_RATE):
    return btchr / ihr


def consumption_per_hash(mpe=MINER_POWER_EFFICIENCY):
    return ((mpe / (60 * 60)) / 1000) / 1000000000000


def consumption_per_day(cph=consumption_per_hash(), ihr=INDIVIDUAL_HASH_RATE):
    return cph * 60 * 60 * 24 * ihr


def watts(cpd=consumption_per_day()):
    return (cpd / 24) * 1000


def costs(el_price, cpd=consumption_per_day()):
    return cpd * el_price


def revenue(btc_price, mhr=miner_hash_ratio(), m=sum_tx_fees(), block_reward=BLOCK_REWARD, reward_in_usd=False):
    # daily revenue
    if reward_in_usd:
        reward = block_reward
    else:
        reward = block_reward * btc_price
    return ((reward + m) / mhr) * 6 * 24  # 6 because every block is mined every 10 minutes


def profit(el_price, btc_price, mhr=miner_hash_ratio(), cpd=consumption_per_day(), m=sum_tx_fees(tpb=TXS_PER_BLOCK, fpt=FEE_PER_TX)):
    return revenue(btc_price, mhr=mhr, m=m) - costs(el_price, cpd=cpd)


def profit2(h, toth):
    btcpr = 19000
    elpr=0.03

    # revenue
    block_reward = 6.25
    mhr = toth / h
    revenue = (block_reward / mhr) * 6 * 24 * btcpr

    # costs
    mpe = 29.55
    cph = ((mpe / (60 * 60)) / 1000) / 1000000000000
    cpd = cph * 60 * 60 * 24 * h
    costs = cpd * elpr

    return revenue - costs


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
