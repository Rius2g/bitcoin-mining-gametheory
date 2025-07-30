import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
import random
import os
import itertools
import time



def debug_all_strategies(df_row, miner_hashrate=90, energy_sector="texas"):
    """
    Debug function to compare all strategies against the same data point,
    with enough decimal places to see the tiny revenue.
    """
    print(f"\n--- DEBUG {energy_sector.upper()} @ {df_row.name} ---")

    # 1) Create the underlying pool strategy
    underlying = PPLNSPool(miner_hashrate, "DebugPool", fee=0.01, efficiency=0.97, energy_sector=energy_sector)

    # 2) Fetch energy price
    energy_price = None
    for col in [f"{energy_sector} price usd kwh", energy_sector.lower() + " price usd kwh"]:
        if col in df_row.index and "dynamic" not in col.lower():
            energy_price = df_row[col]
            break
    if energy_price is None:
        energy_price = 0.07  # fallback default

    # 3) Compute energy cost
    energy_cost = calculate_energy_cost(miner_hashrate, energy_price)
    print(f" energy_price      = {energy_price:.6f} USD/kWh")
    print(f" energy_cost       = {energy_cost:.6f} USD")

    # 4) Compute underlying BTC reward
    underlying_reward = underlying.calculate_reward(df_row)
    print(f" underlying_reward = {underlying_reward:.10f} BTC")

    # 5) Compute revenue & profit
    btc_price      = df_row["BTC market price usd"]
    mining_revenue = underlying_reward * btc_price
    potential_profit = mining_revenue - energy_cost

    # 6) Print both fixed-width and scientific to capture tiny values
    print(f" mining_revenue    = {mining_revenue:.10f} USD  ({mining_revenue:.3e} USD)")
    print(f" potential_profit  = {potential_profit:.10f} USD  ({potential_profit:.3e} USD)")
    print(f" profitable?       = {potential_profit > 0}")


# ============================================================================
# Enhanced Mining Simulation Script
# This script simulates the profitability of different miner sizes with different
# strategies (always-on, stop-loss, linear diminishing, step diminishing)
# across different energy sectors and underlying mining pool strategies
# ============================================================================

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================

def load_data(file_path):
    """
    Load the Excel dataset and prepare it for analysis.
    """
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    # Ensure timestamp is in datetime format and set as index
    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    if 'timestamp' in df.columns:
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
    
    # Print column information to help debugging
    
   
    return df

# ============================================================================
# MINING COST CALCULATION
# ============================================================================

def calculate_energy_cost(hashrate_th, energy_price_per_kwh,
                          efficiency_j_per_th=18.6, hours=3):
    # energy in J = eff(J/TH) * TH * (3600 * hours)
    # kWh    = J / 3_600_000
    # USD    = kWh * USD/kWh
    joules = efficiency_j_per_th * hashrate_th * 3600 * hours
    kwh    = joules / 3_600_000
    return kwh * energy_price_per_kwh

# ============================================================================
# BASE MINING STRATEGY CLASS
# ============================================================================

class MiningStrategy:
    def __init__(self, name, miner_hashrate, energy_sector="texas"):
        self.name = name
        self.miner_hashrate = miner_hashrate  # in TH/s
        self.energy_sector = energy_sector
        self.rewards = []
        self.timestamps = []
        self.cumulative_rewards = []
        self.costs = []
        self.profits = []
        self.cumulative_profits = []
        self.hashrates_used = []
        self.total_btc = 0
        self.total_profit_usd = 0

    def calculate_reward(self, row):
        """
        Calculate reward for a single time period.
        To be implemented in child classes.
        """
        return 0, 0


    def analyze_profitability(self, row, potential_reward, energy_price, energy_cost, btc_price, log_always=False):
        """
        Analyze and log profitability metrics for any strategy.
        
        Parameters:
        -----------
        row : DataFrame row
            The current time period data
        potential_reward : float
            Potential BTC reward at full hashrate
        energy_price : float
            Energy price in USD/kWh
        energy_cost : float
            Total energy cost in USD
        btc_price : float
            Current BTC price in USD
        log_always : bool
            If True, always log regardless of random chance
        """
        mining_revenue = potential_reward * btc_price
        potential_profit = mining_revenue - energy_cost
        
                
        return potential_profit, mining_revenue


    def get_energy_cost(self, row, actual_hashrate):
        """
        Get the energy cost for the current period based on 
        the energy sector and actual hashrate used.
        """
        # Try different possible column name formats - exclude "Dynnamic exchange rate"
        possible_columns = [
            f"{self.energy_sector} price usd kwh",
            f"{self.energy_sector.lower()} price usd kwh"
        ]
        
        
        # Check for exact column names first, excluding dynamic columns
        for column in possible_columns:
            if column in row.index and "dynnamic" not in column.lower() and "dynamic" not in column.lower():
                energy_price = row[column]
                return calculate_energy_cost(actual_hashrate, energy_price)
        
        # If exact column names not found, try a more general approach
        for column in row.index:
            column_lower = column.lower()
            if (self.energy_sector.lower() in column_lower and 
                "price" in column_lower and 
                "usd" in column_lower and 
                "kwh" in column_lower and 
                "dynnamic" not in column_lower and 
                "dynamic" not in column_lower):
                
                energy_price = row[column]
                return calculate_energy_cost(actual_hashrate, energy_price)
        
        # If we couldn't find the column, use a default price based on energy sector
        default_prices = {
            "denmark": 0.16,
            "texas": 0.07,
            "kazakhstan": 0.05,
            "china": 0.07
        }
        default_price = default_prices.get(self.energy_sector.lower(), 0.07)
        if random.random() < 0.01:  # Only print occasionally
            print(f"Warning: Energy price column for '{self.energy_sector}' not found. Using default price: ${default_price}")
        return calculate_energy_cost(actual_hashrate, default_price)
    
    def calculate_energy_cost(self, row, actual_hashrate):
        """
        Calculate the energy cost for the current period based on 
        the energy sector and actual hashrate used.
        """
        # Try different possible column name formats - exclude "Dynnamic exchange rate"
        possible_columns = [
            f"{self.energy_sector} price usd kwh",
            f"{self.energy_sector.lower()} price usd kwh"
        ]
       
        for column in possible_columns:
            col_lc = column.lower()
            if ('price' in col_lc and 'usd' in col_lc and 'kwh' in col_lc and self.energy_sector.lower() in col_lc ):
                energy_price = row[column]
                return calculate_energy_cost(actual_hashrate, energy_price)
        
        # If we couldn't find the column, use a default price based on energy sector
        default_prices = {
            "denmark": 0.16,
            "texas": 0.07,
            "kazakhstan": 0.05,
            "china": 0.07
        }
        default_price = default_prices.get(self.energy_sector.lower(), 0.07)
        print(f"Warning: Energy price column for '{self.energy_sector}' not found. Using default price: ${default_price}")
        return calculate_energy_cost(actual_hashrate, default_price)
    
    def run_simulation(self, df, sell_strategy="immediate"):
        """
        Run the simulation over all time periods in the provided dataframe.
        
        Parameters:
        sell_strategy : str
            "immediate" - Calculate profit based on BTC price at time of mining
            "hold" - Calculate profit based on final period BTC price
        """
        # Initialize lists
        rewards = []
        timestamps = []
        cumulative_rewards = []
        costs = []
        immediate_profits = []  # Profit if sold immediately
        hold_profits = []  # Profit if held until end
        immediate_cumulative_profits = []
        hold_cumulative_profits = []
        hashrates_used = []
        
        # Reset totals
        self.total_btc = 0
        self.immediate_profit_usd = 0
        self.hold_profit_usd = 0
        immediate_cumulative_profit = 0
        
        # Store all BTC earned with its acquisition price for "hold" strategy calculation
        btc_holdings = []
        
        # Get final BTC price for "hold" strategy
        final_btc_price = df["BTC market price usd"].iloc[-1]
        total_energy_cost = 0
        
        # Process each time period
        for idx, row in df.iterrows():
            # Calculate mining reward and hashrate used for this period
            reward, actual_hashrate = self.calculate_reward(row)
            energy_cost = self.get_energy_cost(row, actual_hashrate)
            total_energy_cost += energy_cost
            
            # Current BTC price
            btc_price = row["BTC market price usd"]
            
            # Store BTC earned with its price for later calculation
            if reward > 0:
                btc_holdings.append((reward, btc_price))
            
            # Calculate immediate profit (if sold at current price)
            immediate_profit = (reward * btc_price) - energy_cost
            
            # Update tracking lists
            rewards.append(reward)
            timestamps.append(idx)
            self.total_btc += reward
            cumulative_rewards.append(self.total_btc)
            costs.append(energy_cost)
            immediate_profits.append(immediate_profit)
            immediate_cumulative_profit += immediate_profit
            immediate_cumulative_profits.append(immediate_cumulative_profit)
            hashrates_used.append(actual_hashrate)
        
        # For "hold" strategy, calculate profit if all BTC is sold at final price
        hold_profit = self.total_btc * final_btc_price - total_energy_cost
            
        # Choose which profit calculation to return based on strategy
        if sell_strategy == "immediate":
            profits = immediate_profits
            cumulative_profits = immediate_cumulative_profits
            final_profit = immediate_cumulative_profit
        else:  # "hold"
            profits = [0] * len(immediate_profits)  # Placeholder
            profits[-1] = hold_profit  # Only record profit at the end
            cumulative_profits = [0] * (len(immediate_profits) - 1) + [hold_profit]
            final_profit = hold_profit
        
        # Create result DataFrame with the original timestamps as index
        result_data = {
            'reward': rewards,
            'cumulative_reward': cumulative_rewards,
            'energy_cost': costs,
            'profit_usd': profits,
            'cumulative_profit_usd': cumulative_profits,
            'hashrate_used': hashrates_used,
            'immediate_profit': immediate_profits,
            'hold_profit': [0] * (len(immediate_profits) - 1) + [hold_profit]
        }
        
        result_df = pd.DataFrame(result_data, index=timestamps)
        
        # Check if the index is DatetimeIndex, if not try to convert it
        if not isinstance(result_df.index, pd.DatetimeIndex):
            try:
                result_df.index = pd.to_datetime(result_df.index)
            except:
                print(f"WARNING: Could not convert index to DatetimeIndex for {self.name}")
        
        return result_df

# ============================================================================
# MAIN STRATEGIES (Always-on, Stop-loss, Linear diminishing, Step diminishing)
# ============================================================================

class AlwaysOnStrategy(MiningStrategy):
    """Base class for always-on mining strategy (100% mining)"""
    def __init__(self, name, miner_hashrate, energy_sector="texas"):
        super().__init__(f"{name} (Always-On)", miner_hashrate, energy_sector)
        self.underlying_strategy = None
    
    def set_underlying_strategy(self, strategy):
        """Set the underlying mining pool strategy"""
        self.underlying_strategy = strategy
    
    def calculate_reward(self, row):
        """
        Always mine at full capacity, regardless of profitability.
        Returns reward and actual hashrate used.
        """
        reward = self.underlying_strategy.calculate_reward(row)
        return reward, self.miner_hashrate  # Always use full hashrate

class StopLossStrategy(MiningStrategy):
    """Stop mining completely when not profitable"""
    def __init__(self, name, miner_hashrate, energy_sector="texas"):
        super().__init__(f"{name} (Stop-Loss)", miner_hashrate, energy_sector)
        self.underlying_strategy = None
        self.profitable_periods = 0
        self.total_periods = 0

    def set_underlying_strategy(self, strategy):
        """Set the underlying mining pool strategy"""
        self.underlying_strategy = strategy
    
    def calculate_reward(self, row):
        """
        Check if mining would be profitable. If not, turn off mining.
        Returns reward and actual hashrate used.
        """
        self.total_periods += 1
        
        # First, calculate potential reward at full hashrate
        potential_reward = self.underlying_strategy.calculate_reward(row)
        
        # Get energy price
        energy_price = None
        for column in [f"{self.energy_sector} price usd kwh", f"{self.energy_sector.lower()} price usd kwh"]:
            if column in row.index and "dynnamic" not in column.lower() and "dynamic" not in column.lower():
                energy_price = row[column]
                break
        
        if energy_price is None:
            # Use default if column not found
            default_prices = {"denmark": 0.16, "texas": 0.07, "kazakhstan": 0.05, "china": 0.07}
            energy_price = default_prices.get(self.energy_sector.lower(), 0.07)
        
        # Calculate energy cost
        energy_cost = calculate_energy_cost(self.miner_hashrate, energy_price)
        
        # Calculate potential profit (in USD)
        btc_price = row["BTC market price usd"]
        
        # Analyze profitability
        potential_profit, _ = self.analyze_profitability(
            row, potential_reward, energy_price, energy_cost, btc_price)
        
        if potential_profit > 0:
            # If profitable, mine at full capacity
            self.profitable_periods += 1
            return potential_reward, self.miner_hashrate
        else:
            # If not profitable, don't mine at all
            return 0, 0


class LinearDiminishingStrategy(MiningStrategy):
    """Linearly reduce hashrate based on degree of unprofitability"""
    def __init__(self, name, miner_hashrate, energy_sector="texas", min_hashrate_pct=0.1):
        super().__init__(f"{name} (Linear-Diminish)", miner_hashrate, energy_sector)
        self.underlying_strategy = None
        self.min_hashrate_pct = min_hashrate_pct  # Minimum hashrate as % of full capacity
    
    def set_underlying_strategy(self, strategy):
        """Set the underlying mining pool strategy"""
        self.underlying_strategy = strategy
    
    def calculate_reward(self, row):
        """
        Linearly reduce hashrate based on how unprofitable mining is.
        Returns reward and actual hashrate used.
        """
        # Calculate potential reward and cost at full hashrate
        potential_reward = self.underlying_strategy.calculate_reward(row)
        
        # Try different possible column name formats for energy price - exclude dynamic exchange rate
        energy_price = None
        for column in [f"{self.energy_sector} price usd kwh", f"{self.energy_sector.lower()} price usd kwh"]:
            col_lc = column.lower()
            if ('price' in col_lc and 'usd' in col_lc and 'kwh' in col_lc and self.energy_sector.lower() in col_lc ):
               energy_price = row[column]
               break
        
        if energy_price is None:
            # Use default if column not found
            default_prices = {"denmark": 0.16, "texas": 0.07, "kazakhstan": 0.05, "china": 0.07}
            energy_price = default_prices.get(self.energy_sector.lower(), 0.07)
            
        full_energy_cost = calculate_energy_cost(self.miner_hashrate, energy_price)
        
        # Calculate potential profit (in USD)
        btc_price = row["BTC market price usd"]
        potential_profit = (potential_reward * btc_price) - full_energy_cost
        
        if potential_profit >= 0:
            # If profitable, mine at full capacity
            return potential_reward, self.miner_hashrate
        else:
            # If not profitable, calculate how unprofitable it is
            # as a percentage of mining revenue
            mining_revenue = potential_reward * btc_price
            if mining_revenue > 0:
                loss_ratio = -potential_profit / mining_revenue
                
                # Calculate new hashrate (linear reduction)
                # The more unprofitable, the lower the hashrate
                reduction_factor = max(1 - loss_ratio, self.min_hashrate_pct)
                actual_hashrate = self.miner_hashrate * reduction_factor
                
                # Calculate adjusted reward based on reduced hashrate
                # assuming a linear relationship between hashrate and reward
                adjusted_reward = potential_reward * (actual_hashrate / self.miner_hashrate)
                
                return adjusted_reward, actual_hashrate
            else:
                # If no mining revenue, use minimum hashrate
                actual_hashrate = self.miner_hashrate * self.min_hashrate_pct
                adjusted_reward = potential_reward * self.min_hashrate_pct
                return adjusted_reward, actual_hashrate

class StepDiminishingStrategy(MiningStrategy):
    """Reduce hashrate in steps based on unprofitability thresholds"""
    def __init__(self, name, miner_hashrate, energy_sector="texas"):
        super().__init__(f"{name} (Step-Diminish)", miner_hashrate, energy_sector)
        self.underlying_strategy = None
        
        # Define hashrate reduction steps based on profitability
        # (loss_ratio, hashrate_percentage)
        self.steps = [
            (0.0, 1.0),    # No loss -> 100% hashrate
            (0.2, 0.8),    # 0-20% loss -> 80% hashrate
            (0.4, 0.6),    # 20-40% loss -> 60% hashrate
            (0.6, 0.4),    # 40-60% loss -> 40% hashrate
            (0.8, 0.2),    # 60-80% loss -> 20% hashrate
            (1.0, 0.1),    # >80% loss -> 10% hashrate
        ]
    
    def set_underlying_strategy(self, strategy):
        """Set the underlying mining pool strategy"""
        self.underlying_strategy = strategy
    
    def calculate_reward(self, row):
        """
        Reduce hashrate based on step thresholds of unprofitability.
        Returns reward and actual hashrate used.
        """
        # Calculate potential reward and cost at full hashrate
        potential_reward = self.underlying_strategy.calculate_reward(row)
        
        # Try different possible column name formats for energy price - exclude dynamic exchange rate
        energy_price = None
        for column in [f"{self.energy_sector} price usd kwh", f"{self.energy_sector.lower()} price usd kwh"]:
            col_lc = column.lower()
            if ('price' in col_lc and 'usd' in col_lc and 'kwh' in col_lc and self.energy_sector.lower() in col_lc ):
                energy_price = row[column]
                break
        
        if energy_price is None:
            # Use default if column not found
            default_prices = {"denmark": 0.16, "texas": 0.07, "kazakhstan": 0.05, "china": 0.07}
            energy_price = default_prices.get(self.energy_sector.lower(), 0.07)
            
        full_energy_cost = calculate_energy_cost(self.miner_hashrate, energy_price)
        
        # Calculate potential profit (in USD)
        btc_price = row["BTC market price usd"]
        potential_profit = (potential_reward * btc_price) - full_energy_cost
        
        if potential_profit >= 0:
            # If profitable, mine at full capacity
            return potential_reward, self.miner_hashrate
        else:
            # If not profitable, calculate how unprofitable it is
            # as a percentage of mining revenue
            mining_revenue = potential_reward * btc_price
            
            if mining_revenue > 0:
                loss_ratio = -potential_profit / mining_revenue
                
                # Find the appropriate step for this loss ratio
                hashrate_pct = self.steps[-1][1]  # Default to minimum
                for threshold, pct in self.steps:
                    if loss_ratio <= threshold:
                        hashrate_pct = pct
                        break
                
                actual_hashrate = self.miner_hashrate * hashrate_pct
                adjusted_reward = potential_reward * hashrate_pct
                
                return adjusted_reward, actual_hashrate
            else:
                # If no mining revenue, use minimum hashrate
                hashrate_pct = self.steps[-1][1]
                actual_hashrate = self.miner_hashrate * hashrate_pct
                adjusted_reward = 0
                return adjusted_reward, actual_hashrate

# ============================================================================
# ORIGINAL POOL STRATEGIES (Underlying Strategies)
# ============================================================================

class SoloMining(MiningStrategy):
    """
    Solo mining strategy: trying to find blocks independently.
    """
    def __init__(self, miner_hashrate, energy_sector="texas"):
        super().__init__("Solo Mining", miner_hashrate, energy_sector)
        self.blocks_found = 0
        self.block_timestamps = []
        self.block_rewards = []
    
    def calculate_reward(self, row):
        network_hashrate = row['hash_rate'] 
        p_block = self.miner_hashrate / network_hashrate
        blocks_in_period = 18  # 3 hours / 10-minute target (18 blocks)
        blocks_found = np.random.binomial(blocks_in_period, p_block)
        if blocks_found > 0:
            self.blocks_found += blocks_found
            self.block_timestamps.append(row.name)
            block_reward = row['Block reward (BTC)'] + row['transaction fees btc']
            self.block_rewards.append(block_reward * blocks_found)
            return blocks_found * block_reward
        else:
            return 0

class FPPSPool(MiningStrategy):
    """
    FPPS Pool: Pay Per Share pool including transaction fees.
    """
    def __init__(self, miner_hashrate, pool_name, fee=0.02, efficiency=1.0, energy_sector="texas"):
        super().__init__(f"{pool_name} (FPPS)", miner_hashrate, energy_sector)
        self.pool_fee = fee
        self.efficiency = efficiency
    
    def calculate_reward(self, row):
        network_hashrate = row['hash_rate'] 
        hashrate_share = self.miner_hashrate / network_hashrate
        blocks_in_period = 18
        block_reward = row['Block reward (BTC)'] + row['transaction fees btc']
        expected_reward = hashrate_share * blocks_in_period * block_reward * self.efficiency
        return expected_reward * (1 - self.pool_fee)

class PPLNSPool(MiningStrategy):
    """
    PPLNS Pool: Payment based on shares over a moving window.
    """
    def __init__(self, miner_hashrate, pool_name, fee=0.01, efficiency=0.97, energy_sector="texas"):
        super().__init__(f"{pool_name} (PPLNS)", miner_hashrate, energy_sector)
        self.pool_fee = fee
        self.efficiency = efficiency  # Efficiency factor for PPLNS pools.
        self.consistency_factor = 1.0  # Simulate loyalty benefits.
    
    def calculate_reward(self, row):
        network_hashrate = row['hash_rate'] 
        hashrate_share = self.miner_hashrate / network_hashrate
        blocks_in_period = 18
        block_reward = row['Block reward (BTC)'] + row['transaction fees btc']
        expected_reward = hashrate_share * blocks_in_period * block_reward * self.efficiency * self.consistency_factor
        # Slowly increase the consistency factor, capping at 1.05.
        self.consistency_factor = min(1.05, self.consistency_factor + 0.0001)
        return expected_reward * (1 - self.pool_fee)

class TIDESPool(MiningStrategy):
    """
    TIDES Pool: Time and Difficulty-based Economic Sharing model (e.g., Ocean).
    """
    def __init__(self, miner_hashrate, pool_name, fee=0.0, efficiency=1.0, energy_sector="texas"):
        super().__init__(f"{pool_name} (TIDES)", miner_hashrate, energy_sector)
        self.pool_fee = fee
        self.efficiency = efficiency
        self.consistency_factor = 1.0
    
    def calculate_reward(self, row):
        network_hashrate = row['hash_rate'] 
        hashrate_share = self.miner_hashrate / network_hashrate
        blocks_in_period = 18
        block_reward = row['Block reward (BTC)'] + row['transaction fees btc']
        expected_reward = hashrate_share * blocks_in_period * block_reward * self.efficiency * self.consistency_factor
        self.consistency_factor = min(1.05, self.consistency_factor + 0.0001)
        return expected_reward * (1 - self.pool_fee)

class HybridMining(MiningStrategy):
    """
    Hybrid Mining: Split the miner's hash rate between solo and pool mining.
    """
    def __init__(self, miner_hashrate, solo_ratio=0.5, pool_name="Generic Pool", pool_type="FPPS", pool_fee=0.02, energy_sector="texas"):
        super().__init__(f"Hybrid ({int(solo_ratio*100)}% Solo, {int((1-solo_ratio)*100)}% {pool_name})", miner_hashrate, energy_sector)
        self.solo_ratio = solo_ratio
        self.pool_fee = pool_fee
        
        # Split hashrate between solo mining and pool mining.
        self.solo_hashrate = miner_hashrate * solo_ratio
        self.pool_hashrate = miner_hashrate * (1 - solo_ratio)
        
        self.solo_strategy = SoloMining(self.solo_hashrate, energy_sector)
        
        if pool_type.upper() in ["FPPS", "FPPS+"]:
            self.pool_strategy = FPPSPool(self.pool_hashrate, pool_name, fee=pool_fee, energy_sector=energy_sector)
        elif pool_type.upper() == "PPLNS":
            self.pool_strategy = PPLNSPool(self.pool_hashrate, pool_name, fee=pool_fee, energy_sector=energy_sector)
        else:
            self.pool_strategy = FPPSPool(self.pool_hashrate, pool_name, fee=pool_fee, energy_sector=energy_sector)
    
    def calculate_reward(self, row):
        solo_reward = self.solo_strategy.calculate_reward(row)
        pool_reward = self.pool_strategy.calculate_reward(row)
        return solo_reward + pool_reward

# ============================================================================
# STRATEGY CREATION & SIMULATION
# ============================================================================

def create_underlying_strategies(miner_hashrate, miner_size, energy_sector="texas"):
    """
    Create a list of underlying mining pool strategies based on miner size.
    The energy_sector parameter is used for initialization.
    """
    strategies = []
    
    # Use "texas" as default energy sector for initial setup if none provided
    energy_sector = energy_sector if energy_sector else "texas"
    
    if miner_size == "Small":
        strategies = [
            TIDESPool(miner_hashrate, "Ocean", fee=0.0, energy_sector=energy_sector),
            PPLNSPool(miner_hashrate, "Antpool", fee=0.0, efficiency=0.97, energy_sector=energy_sector),
            PPLNSPool(miner_hashrate, "SECPOOL", fee=0.0, efficiency=0.97, energy_sector=energy_sector),
            PPLNSPool(miner_hashrate, "Spiderpool", fee=0.0, efficiency=0.97, energy_sector=energy_sector),
            PPLNSPool(miner_hashrate, "SBI Crypto", fee=0.005, efficiency=0.97, energy_sector=energy_sector)
        ]
    elif miner_size == "Medium":
        strategies = [
            SoloMining(miner_hashrate, energy_sector=energy_sector),
            TIDESPool(miner_hashrate, "Ocean", fee=0.0, energy_sector=energy_sector),
            HybridMining(miner_hashrate, solo_ratio=0.3, pool_name="ViaBTC", pool_type="PPLNS", pool_fee=0.02, energy_sector=energy_sector),
            HybridMining(miner_hashrate, solo_ratio=0.3, pool_name="Foundry USA", pool_type="FPPS", pool_fee=0.0, energy_sector=energy_sector),
            PPLNSPool(miner_hashrate, "Antpool", fee=0.0, efficiency=0.97, energy_sector=energy_sector)
        ]
    elif miner_size == "Large":
        strategies = [
            SoloMining(miner_hashrate, energy_sector=energy_sector),
            TIDESPool(miner_hashrate, "Ocean", fee=0.0, energy_sector=energy_sector),
            PPLNSPool(miner_hashrate, "Antpool", fee=0.0, efficiency=0.97, energy_sector=energy_sector),
            PPLNSPool(miner_hashrate, "SECPOOL", fee=0.0, efficiency=0.97, energy_sector=energy_sector),
            PPLNSPool(miner_hashrate, "Spiderpool", fee=0.0, efficiency=0.97, energy_sector=energy_sector)
        ]
    elif miner_size == "Industrial":
        strategies = [
            TIDESPool(miner_hashrate, "Ocean", fee=0.0, energy_sector=energy_sector),
            SoloMining(miner_hashrate, energy_sector=energy_sector),
            PPLNSPool(miner_hashrate, "Antpool", fee=0.0, efficiency=0.97, energy_sector=energy_sector),
            PPLNSPool(miner_hashrate, "SECPOOL", fee=0.0, efficiency=0.97, energy_sector=energy_sector),
            PPLNSPool(miner_hashrate, "Spiderpool", fee=0.0, efficiency=0.97, energy_sector=energy_sector)
        ]
    
    return strategies

def create_main_strategies(underlying_strategy, energy_sector):
    """
    Create main strategies (Always-On, Stop-Loss, Linear/Step Diminishing) using an underlying strategy.
    """
    miner_hashrate = underlying_strategy.miner_hashrate
    name = underlying_strategy.name
    
    # Create the four main strategy types
    always_on = AlwaysOnStrategy(name, miner_hashrate, energy_sector)
    always_on.set_underlying_strategy(underlying_strategy)
    
    stop_loss = StopLossStrategy(name, miner_hashrate, energy_sector)
    stop_loss.set_underlying_strategy(underlying_strategy)
    
    linear_diminish = LinearDiminishingStrategy(name, miner_hashrate, energy_sector)
    linear_diminish.set_underlying_strategy(underlying_strategy)
    
    step_diminish = StepDiminishingStrategy(name, miner_hashrate, energy_sector)
    step_diminish.set_underlying_strategy(underlying_strategy)
    
    return [always_on, stop_loss, linear_diminish, step_diminish]

def run_mining_simulation(file_path, miner_size, num_iterations=10, random_seed_start=42):
    """
    Run the mining simulation for the given miner size, testing all combinations
    of main strategies, underlying strategies, and energy sectors.
    """
    # Load data
    df = load_data(file_path)
    
    # Define miner hashrates based on size
    miner_sizes = {
            "Small": 180,
            "Medium": 540,
            "Large": 1350,
            "Industrial": 100000
        }
    miner_hashrate = miner_sizes.get(miner_size, 10000)    
    # Define energy sectors
    energy_sectors = ["denmark", "texas", "kazakhstan", "china"]
    
    # Prepare to store all results
    all_strategies = []
    all_combined_results = []
    
    # Calculate total simulations for progress tracking
    pool_strategies_count = len(create_underlying_strategies(miner_hashrate, miner_size))
    main_strategies_per_pool = 4  # Always-on, Stop-loss, Linear-dim, Step-dim
    total_strategies = pool_strategies_count * len(energy_sectors) * main_strategies_per_pool
    total_simulations = total_strategies * num_iterations
    simulations_completed = 0
    start_time = time.time()
    
    # Run all combinations for each iteration
    for i in range(num_iterations):
        seed = random_seed_start + i 
        np.random.seed(seed) 
        random.seed(seed)
        
        print(f"Iteration {i+1}/{num_iterations} with seed {seed}")
        
        # For this iteration, collect results for all combinations
        iteration_results = {}
        
        # For each energy sector and underlying strategy combination
        for energy_sector in energy_sectors:
            base_strategies = create_underlying_strategies(miner_hashrate, miner_size, energy_sector)
            
            for strat_class in base_strategies:
                # Create main strategies based on this underlying strategy
                main_strategies = create_main_strategies(strat_class, energy_sector)
                
                # Save reference to all strategies for the first iteration
                if i == 0:
                    all_strategies.extend(main_strategies)
                
                # Run simulation for each main strategy
                for strategy in main_strategies:
                    result_df = strategy.run_simulation(df)
                    
                    # Store results with a unique key that includes strategy info
                    strategy_key = f"{strategy.name}_{energy_sector}"
                    iteration_results[strategy_key] = result_df
                    
                    # Update progress tracking
                    simulations_completed += 1
                    if simulations_completed % 5 == 0:  # Update every 5 simulations
                        elapsed_time = time.time() - start_time
                        progress_pct = simulations_completed / total_simulations * 100
                        avg_time_per_sim = elapsed_time / simulations_completed
                        remaining_sims = total_simulations - simulations_completed
                        est_remaining_time = remaining_sims * avg_time_per_sim
                        
                        print(f"Progress: {simulations_completed}/{total_simulations} simulations " +
                              f"({progress_pct:.1f}%) | " +
                              f"Est. time remaining: {est_remaining_time/60:.1f} minutes")
        
        # Create a dictionary to store all columns before creating the DataFrame
        combined_data = {}
        for strategy_key, result_df in iteration_results.items():
            # Extract metrics and add to dictionary
            combined_data[f"{strategy_key}_reward"] = result_df["reward"]
            combined_data[f"{strategy_key}_cumulative"] = result_df["cumulative_reward"]
            combined_data[f"{strategy_key}_profit"] = result_df["profit_usd"]
            combined_data[f"{strategy_key}_cumulative_profit"] = result_df["cumulative_profit_usd"]
            combined_data[f"{strategy_key}_hashrate"] = result_df["hashrate_used"]
        
        # Create DataFrame at once to avoid fragmentation
        combined_results = pd.DataFrame(combined_data, index=df.index)
        all_combined_results.append(combined_results)
        
        # Force garbage collection to free memory
        combined_data = None
        iteration_results = None
        import gc
        gc.collect()
    
    # Average the results from all iterations
    print(f"Calculating average results across {num_iterations} iterations...")
    averaged_results = average_simulation_results(all_combined_results, all_strategies, energy_sectors)
    print("\nProfitability Analysis by Region:")
    for strategy in all_strategies:
        if isinstance(strategy, StopLossStrategy):
            profit_pct = (strategy.profitable_periods / max(1, strategy.total_periods)) * 100
            print(f"{strategy.name} - {strategy.energy_sector}: " + 
                  f"{strategy.profitable_periods}/{strategy.total_periods} " + 
                  f"profitable periods ({profit_pct:.1f}%)")
    
    return averaged_results, df, all_strategies, energy_sectors


def average_simulation_results(all_combined_results, all_strategies, energy_sectors):
    """
    Average the results across all iterations
    """
    first_df = all_combined_results[0] 
    
    # Get all unique strategy keys from the first iteration
    strategy_keys = set()
    for col in first_df.columns:
        # Extract the strategy key (everything before the last underscore and metric name)
        parts = col.split('_')
        if len(parts) >= 3:  # Ensure we have at least strategy_name_metric
            metric = parts[-1]
            if metric in ['reward', 'cumulative', 'profit', 'cumulative_profit', 'hashrate']:
                strategy_key = '_'.join(parts[:-1])
                strategy_keys.add(strategy_key)
    
    # Prepare data for all metrics at once to avoid fragmentation
    averaged_data = {}
    
    # Average each metric for each strategy across all iterations
    for strategy_key in strategy_keys:
        # Average rewards
        reward_cols = [df.get(f"{strategy_key}_reward", pd.Series(0, index=df.index)) 
                      for df in all_combined_results]
        averaged_data[f"{strategy_key}_reward"] = sum(reward_cols) / len(reward_cols)
        
        # Average cumulative rewards
        cumul_cols = [df.get(f"{strategy_key}_cumulative", pd.Series(0, index=df.index)) 
                     for df in all_combined_results]
        averaged_data[f"{strategy_key}_cumulative"] = sum(cumul_cols) / len(cumul_cols)
        
        # Average profits
        profit_cols = [df.get(f"{strategy_key}_profit", pd.Series(0, index=df.index)) 
                      for df in all_combined_results]
        averaged_data[f"{strategy_key}_profit"] = sum(profit_cols) / len(profit_cols)
        
        # Average cumulative profits
        cumul_profit_cols = [df.get(f"{strategy_key}_cumulative_profit", pd.Series(0, index=df.index)) 
                           for df in all_combined_results]
        averaged_data[f"{strategy_key}_cumulative_profit"] = sum(cumul_profit_cols) / len(cumul_profit_cols)
        
        # Average hashrate used
        hashrate_cols = [df.get(f"{strategy_key}_hashrate", pd.Series(0, index=df.index)) 
                        for df in all_combined_results]
        averaged_data[f"{strategy_key}_hashrate"] = sum(hashrate_cols) / len(hashrate_cols)
    
    # Create averaged results DataFrame all at once to avoid fragmentation
    return pd.DataFrame(averaged_data, index=first_df.index)


# ============================================================================
# ANALYSIS & VISUALIZATION FUNCTIONS
# ============================================================================


def create_comparison_plots(summary_df, averaged_results, miner_size):
    """
    Create various visualization plots to compare different strategies.
    """
    print("  Creating top 10 USD profit comparison...")
    # 1. Top 10 strategies by Final USD Profit
    plt.figure(figsize=(14, 8))
    top10 = summary_df.head(10)
    top10_labels = [f"{row['Underlying Strategy']} + {row['Main Strategy']} ({row['Energy Sector']})" 
                   for idx, row in top10.iterrows()]
    
    plt.barh(top10_labels, top10["Final USD Profit"])
    plt.title(f"Top 10 Strategies by USD Profit - {miner_size} Miner")
    plt.xlabel("USD Profit")
    plt.tight_layout()
    plt.savefig(f"equilibrium_plots/{miner_size}_top10_usd_profit.png")
    plt.close()
    
    print("  Creating main strategy comparison...")
    # 2. Comparison of main strategy types (boxplot)
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="Main Strategy", y="Final USD Profit", data=summary_df)
    plt.title(f"USD Profit by Main Strategy Type - {miner_size} Miner")
    plt.tight_layout()
    plt.savefig(f"equilibrium_plots/{miner_size}_main_strategy_comparison.png")
    plt.close()
    
    print("  Creating energy sector comparison...")
    # 3. Comparison by energy sector (boxplot)
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="Energy Sector", y="Final USD Profit", data=summary_df)
    plt.title(f"USD Profit by Energy Sector - {miner_size} Miner")
    plt.tight_layout()
    plt.savefig(f"equilibrium_plots/{miner_size}_energy_sector_comparison.png")
    plt.close()
    
    print("  Creating risk-return plot...")
    # 4. Scatter plot of Risk-Adjusted Return vs Total Profit
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(summary_df["USD Risk-Adjusted Return"], 
              summary_df["Final USD Profit"],
              c=summary_df["Energy Sector"].astype('category').cat.codes,
              s=60, alpha=0.7)
    
    # Add legend for energy sectors
    energy_sectors = summary_df["Energy Sector"].unique()
    legend1 = plt.legend(scatter.legend_elements()[0], 
                        energy_sectors,
                        title="Energy Sector", loc="upper left")
    plt.gca().add_artist(legend1)
    
    plt.title(f"Risk-Adjusted Return vs USD Profit - {miner_size} Miner")
    plt.xlabel("Risk-Adjusted Return (USD)")
    plt.ylabel("Final USD Profit")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"equilibrium_plots/{miner_size}_risk_vs_profit.png")
    plt.close()
    
    print("  Creating hashrate usage analysis...")
    # 5. Hashrate Usage Analysis
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="Main Strategy", y="Avg Hashrate Usage (%)", data=summary_df)
    plt.title(f"Hashrate Usage by Main Strategy - {miner_size} Miner")
    plt.tight_layout()
    plt.savefig(f"equilibrium_plots/{miner_size}_hashrate_usage.png")
    plt.close()
    
    print("  Creating cumulative profit chart...")
    # 6. Plot cumulative profits over time for top 5 strategies
    plt.figure(figsize=(14, 8))
    top5_keys = summary_df.head(5).index
    
    for strategy_key in top5_keys:
        label = f"{summary_df.loc[strategy_key, 'Underlying Strategy']} + {summary_df.loc[strategy_key, 'Main Strategy']} ({summary_df.loc[strategy_key, 'Energy Sector']})"
        plt.plot(averaged_results.index, averaged_results[f"{strategy_key}_cumulative_profit"], label=label)
    
    plt.title(f"Cumulative USD Profit - Top 5 Strategies ({miner_size} Miner)")
    plt.xlabel("Date")
    plt.ylabel("USD")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"equilibrium_plots/{miner_size}_top5_cumulative_usd.png")
    plt.close()
    
    print("  Plots complete for this miner size.")

def compare_across_sizes(all_results):
    """
    Compare results across different miner sizes.
    """
    # Combine all summary data
    combined_summary = pd.concat(all_results, keys=list(all_results.keys()))
    combined_summary = combined_summary.reset_index(level=1, drop=True).reset_index(names='Miner Size')
    
    # Create directory for cross-size comparisons
    if not os.path.exists('equilibrium_plots/cross_size'):
        os.makedirs('equilibrium_plots/cross_size')
    
    # 1. Best Main Strategy by Miner Size
    plt.figure(figsize=(14, 8))
    best_by_size = combined_summary.groupby(['Miner Size', 'Main Strategy'])['Final USD Profit'].mean().reset_index()
    pivot = best_by_size.pivot(index='Main Strategy', columns='Miner Size', values='Final USD Profit')
    pivot.plot(kind='bar', figsize=(14, 8))
    plt.title("Average USD Profit by Main Strategy and Miner Size")
    plt.ylabel("USD Profit")
    plt.legend(title="Miner Size")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("equilibrium_plots/cross_size/main_strategy_by_size.png")
    plt.close()
    
    # 2. Best Energy Sector by Miner Size
    plt.figure(figsize=(14, 8))
    best_by_energy = combined_summary.groupby(['Miner Size', 'Energy Sector'])['Final USD Profit'].mean().reset_index()
    pivot = best_by_energy.pivot(index='Energy Sector', columns='Miner Size', values='Final USD Profit')
    pivot.plot(kind='bar', figsize=(14, 8))
    plt.title("Average USD Profit by Energy Sector and Miner Size")
    plt.ylabel("USD Profit")
    plt.legend(title="Miner Size")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("equilibrium_plots/cross_size/energy_sector_by_size.png")
    plt.close()
    
    # 3. Best Underlying Strategy by Miner Size
    plt.figure(figsize=(14, 8))
    # Group by underlying strategy and miner size
    best_underlying = combined_summary.groupby(['Miner Size', 'Underlying Strategy'])['Final USD Profit'].mean().reset_index()
    
    # Get top 3 strategies for each size
    top_strategies = {}
    for size in best_underlying['Miner Size'].unique():
        size_data = best_underlying[best_underlying['Miner Size'] == size]
        top3 = size_data.nlargest(3, 'Final USD Profit')['Underlying Strategy'].values
        top_strategies[size] = list(top3)
    
    # Prepare data for plotting
    sizes = []
    strategies = []
    profits = []
    
    for size, strats in top_strategies.items():
        for strat in strats:
            size_strat_data = best_underlying[(best_underlying['Miner Size'] == size) & 
                                             (best_underlying['Underlying Strategy'] == strat)]
            if not size_strat_data.empty:
                sizes.append(size)
                strategies.append(strat)
                profits.append(size_strat_data['Final USD Profit'].values[0])
    
    # Create DataFrame and plot
    plot_df = pd.DataFrame({
        'Miner Size': sizes,
        'Underlying Strategy': strategies,
        'USD Profit': profits
    })
    
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Miner Size', y='USD Profit', hue='Underlying Strategy', data=plot_df)
    plt.title("Top 3 Underlying Strategies by Miner Size")
    plt.ylabel("USD Profit")
    plt.legend(title="Underlying Strategy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("equilibrium_plots/cross_size/top_underlying_by_size.png")
    plt.close()
    
    # 4. Risk-Adjusted Returns Comparison
    plt.figure(figsize=(14, 8))
    risk_data = combined_summary.groupby(['Miner Size', 'Main Strategy'])['USD Risk-Adjusted Return'].mean().reset_index()
    pivot = risk_data.pivot(index='Main Strategy', columns='Miner Size', values='USD Risk-Adjusted Return')
    pivot.plot(kind='bar', figsize=(14, 8))
    plt.title("Average Risk-Adjusted Return by Main Strategy and Miner Size")
    plt.ylabel("Risk-Adjusted Return")
    plt.legend(title="Miner Size")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("equilibrium_plots/cross_size/risk_adjusted_by_size.png")
    plt.close()
    
    return combined_summary


def analyze_results(averaged_results, df, all_strategies, energy_sectors, miner_size):
    """
    Analyze simulation results and create summary statistics for each strategy combination.
    """
    if not os.path.exists('equilibrium_plots'):
        os.makedirs('equilibrium_plots')
    
    # Check the type of index in averaged_results
    print(f"Index type in averaged_results: {type(averaged_results.index)}")
    is_datetime_index = isinstance(averaged_results.index, pd.DatetimeIndex)
    print(f"Is DatetimeIndex: {is_datetime_index}")
    
    # Create a dictionary to store summary metrics for all strategies
    summary_data = {}
    
    # Progress counter
    total_strategies = len(all_strategies) * len(energy_sectors)
    processed = 0
    
    # Parse strategy names to extract components for analysis
    for strategy in all_strategies:
        for energy_sector in energy_sectors:
            strategy_key = f"{strategy.name}_{energy_sector}"
            
            # Progress indication
            processed += 1
            if processed % 10 == 0:  # Update every 10 strategies
                print(f"Analyzing strategies: {processed}/{total_strategies} ({processed/total_strategies*100:.1f}%)")
            
            # Extract strategy components
            if "(Always-On)" in strategy.name:
                main_strategy_type = "Always-On"
            elif "(Stop-Loss)" in strategy.name:
                main_strategy_type = "Stop-Loss"
            elif "(Linear-Diminish)" in strategy.name:
                main_strategy_type = "Linear-Diminish"
            elif "(Step-Diminish)" in strategy.name:
                main_strategy_type = "Step-Diminish"
            else:
                main_strategy_type = "Unknown"
            
            # Extract underlying strategy name (remove the main strategy type)
            underlying_strategy = strategy.name.replace(f" ({main_strategy_type})", "")
            
            # Calculate summary metrics if this strategy exists in our results
            if f"{strategy_key}_cumulative" in averaged_results.columns:
                # Total BTC mined
                total_btc = averaged_results[f"{strategy_key}_cumulative"].iloc[-1]
                
                # Final USD profit
                final_usd_profit = averaged_results[f"{strategy_key}_cumulative_profit"].iloc[-1]
                
                # Calculate monthly values if we have a DatetimeIndex
                if is_datetime_index:
                    # Calculate monthly BTC mined and USD profits
                    try:
                        monthly_btc = averaged_results[f"{strategy_key}_reward"].resample('ME').sum()
                        monthly_usd_profits = averaged_results[f"{strategy_key}_profit"].resample('ME').sum()
                        
                        # Calculate risk-adjusted returns (Sharpe ratio-like)
                        btc_mean = monthly_btc.mean()
                        btc_std = monthly_btc.std() if monthly_btc.std() > 0 else 1e-10
                        btc_risk_adjusted = btc_mean / btc_std
                        
                        usd_mean = monthly_usd_profits.mean()
                        usd_std = monthly_usd_profits.std() if monthly_usd_profits.std() > 0 else 1e-10
                        usd_risk_adjusted = usd_mean / usd_std
                    except Exception as e:
                        print(f"Error in monthly calculations: {str(e)}")
                        # Use simple averages as fallback
                        btc_mean = averaged_results[f"{strategy_key}_reward"].mean()
                        usd_mean = averaged_results[f"{strategy_key}_profit"].mean()
                        btc_risk_adjusted = 0.0
                        usd_risk_adjusted = 0.0
                else:
                    # If not a DatetimeIndex, use simple averages instead
                    print(f"Using simple averages for {strategy_key} (no DatetimeIndex)")
                    btc_mean = averaged_results[f"{strategy_key}_reward"].mean()
                    usd_mean = averaged_results[f"{strategy_key}_profit"].mean()
                    btc_risk_adjusted = 0.0
                    usd_risk_adjusted = 0.0
                
                # Calculate average hashrate usage (as % of full capacity)
                avg_hashrate_pct = averaged_results[f"{strategy_key}_hashrate"].mean() / strategy.miner_hashrate
                
                # Store summary data
                summary_data[strategy_key] = {
                    "Main Strategy": main_strategy_type,
                    "Underlying Strategy": underlying_strategy,
                    "Energy Sector": energy_sector,
                    "Total BTC Mined": total_btc,
                    "Final USD Profit": final_usd_profit,
                    "Monthly BTC (avg)": btc_mean,
                    "Monthly USD Profit (avg)": usd_mean,
                    "BTC Risk-Adjusted Return": btc_risk_adjusted,
                    "USD Risk-Adjusted Return": usd_risk_adjusted,
                    "Avg Hashrate Usage (%)": avg_hashrate_pct * 100
                }
    
    # Convert to DataFrame for easier analysis
    summary_df = pd.DataFrame.from_dict(summary_data, orient='index')
    
    # Sort by USD Profit for overall ranking
    summary_df = summary_df.sort_values(by="Final USD Profit", ascending=False)
    
    # Save summary to CSV
    summary_df.to_csv(f"equilibrium_plots/{miner_size}_summary.csv")
    
    # Create various comparison plots
    print("Creating comparison plots...")
    create_comparison_plots(summary_df, averaged_results, miner_size)
    
    return summary_df

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main(num_iterations=5):
    file_path = 'bitcoin_metrics_and_energy_updated_final_manually_scaled.xlsx'

    test_df = pd.read_excel(file_path)
    # pick a single timestamp to test
    test_row = test_df.iloc[0]
    for sector in ["denmark","texas","kazakhstan","china"]:
        debug_all_strategies(test_row, miner_hashrate=90, energy_sector=sector)


    miner_sizes = ["Small", "Medium", "Large", "Industrial"]
    
    all_results = {}
    
    for miner_size in miner_sizes:
        print(f"\n{'='*80}")
        print(f"Running analysis for {miner_size} miner")
        print(f"{'='*80}")
        
        try:
            # Load data first to verify file access
            print(f"Successfully loaded data file with {len(test_df)} rows")
            
            # Verify energy price columns exist - filter out the "Dynnamic exchange rate" column
            energy_columns = [col for col in test_df.columns if 'price usd kwh' in col.lower() 
                              and 'dynamic exchange rate'  not in col.lower()]
                             
            print(f"Found energy price columns: {energy_columns}")
            
            if not energy_columns:
                print("WARNING: No energy price columns found! Check your Excel file.")
            
            # Run simulation with progress tracking
            print(f"Starting simulation with {num_iterations} iterations...")
            
            # Calculate total number of combinations for progress indicator
            pool_strategies_count = len(create_underlying_strategies(1_000_0000, miner_size))
            energy_sectors_count = 4  # denmark, texas, kazakhstan, china
            main_strategies_count = 4  # Always-on, Stop-loss, Linear-dim, Step-dim
            total_combinations = pool_strategies_count * energy_sectors_count * main_strategies_count
            
            print(f"Total strategy combinations: {total_combinations}")
            print(f"Total simulation runs: {total_combinations * num_iterations}")
            
            averaged_results, original_df, strategies, energy_sectors = run_mining_simulation(
                file_path, miner_size, num_iterations)
            
            print(f"Analyzing results for {len(strategies)} strategies across {len(energy_sectors)} energy sectors...")
            summary_df = analyze_results(averaged_results, original_df, strategies, energy_sectors, miner_size)
            
            print(f"\nTop 10 strategy combinations for {miner_size} miner:")
            top10 = summary_df[['Main Strategy', 'Underlying Strategy', 'Energy Sector', 'Final USD Profit']].head(10)
            print(top10)
            
            all_results[miner_size] = summary_df
            
        except Exception as e:
            print(f"ERROR processing {miner_size} miner: {str(e)}")
            import traceback
            traceback.print_exc()
            print("Continuing with next miner size...")
            continue
            continue
    
    if len(all_results) > 0:
        # Compare results across different miner sizes
        try:
            print("\nComparing results across all miner sizes...")
            cross_size_summary = compare_across_sizes(all_results)
            print("Cross-size comparison complete.")
        except Exception as e:
            print(f"ERROR in cross-size comparison: {str(e)}")
            import traceback
            traceback.print_exc()
            cross_size_summary = None
    else:
        print("No results to compare across sizes.")
        cross_size_summary = None
    
    print("\n===============================================================")
    print("ANALYSIS COMPLETE - Results saved to 'equilibrium_plots' directory")
    print("===============================================================")
    
    return all_results, cross_size_summary

if __name__ == "__main__":
    # Start with 5 iterations for initial testing
    # Once confirmed working, increase to desired number (10 for testing, 1_000_000 for production)
    all_results, cross_size_summary = main(num_iterations=1000)
