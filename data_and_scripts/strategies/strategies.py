import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
from tqdm import tqdm
import random
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================

def load_data(file_path):
    """
    Load the Excel dataset and prepare it for analysis.
    Assumes a column "energy_cost" may be present for green mining.
    """
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    # Ensure timestamp is in datetime format and set as index.
    if not pd.api.types.is_datetime64_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    # Fill missing values in numeric columns (forward fill, then use column mean).
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].ffill().fillna(df[col].mean())
    
    avg_hashrate = df['hash_rate'].mean()
    print(f"Average network hashrate: {avg_hashrate:.2f} EH/s")
    print(f"Min network hashrate: {df['hash_rate'].min():.2f} EH/s")
    print(f"Max network hashrate: {df['hash_rate'].max():.2f} EH/s")
    
    return df

# ============================================================================
# BASE MINING STRATEGY CLASS
# ============================================================================

class MiningStrategy:
    def __init__(self, name, miner_hashrate):
        self.name = name
        self.miner_hashrate = miner_hashrate  # in TH/s
        self.rewards = []
        self.timestamps = []
        self.cumulative_rewards = []
        self.total_btc = 0

    def calculate_reward(self, row):
        """
        Calculate reward for a single time period.
        To be implemented in child classes.
        """
        pass

    def run_simulation(self, df):
        """
        Run the simulation over all time periods in the provided dataframe.
        """
        self.rewards = []
        self.timestamps = []
        self.cumulative_rewards = []
        self.total_btc = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Simulating {self.name}"):
            reward = self.calculate_reward(row)
            self.rewards.append(reward)
            self.timestamps.append(idx)
            self.total_btc += reward
            self.cumulative_rewards.append(self.total_btc)
        
        return pd.DataFrame({
            'timestamp': self.timestamps,
            'reward': self.rewards,
            'cumulative_reward': self.cumulative_rewards
        }).set_index('timestamp')

# ============================================================================
# ORIGINAL STRATEGIES
# ============================================================================

class SoloMining(MiningStrategy):
    """
    Solo mining strategy: trying to find blocks independently.
    """
    def __init__(self, miner_hashrate):
        super().__init__("Solo Mining", miner_hashrate)
        self.blocks_found = 0
        self.block_timestamps = []
        self.block_rewards = []
    
    def calculate_reward(self, row):
        network_hashrate = row['hash_rate'] * 1000  # Convert EH/s to TH/s
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
    def __init__(self, miner_hashrate, pool_name, fee=0.02, efficiency=1.0):
        super().__init__(f"{pool_name} (FPPS)", miner_hashrate)
        self.pool_fee = fee
        self.efficiency = efficiency
    
    def calculate_reward(self, row):
        network_hashrate = row['hash_rate'] * 1000  # Convert EH/s to TH/s
        hashrate_share = self.miner_hashrate / network_hashrate
        blocks_in_period = 18
        block_reward = row['Block reward (BTC)'] + row['transaction fees btc']
        expected_reward = hashrate_share * blocks_in_period * block_reward * self.efficiency
        return expected_reward * (1 - self.pool_fee)

class PPLNSPool(MiningStrategy):
    """
    PPLNS Pool: Payment based on shares over a moving window.
    """
    def __init__(self, miner_hashrate, pool_name, fee=0.01, efficiency=0.97):
        super().__init__(f"{pool_name} (PPLNS)", miner_hashrate)
        self.pool_fee = fee
        self.efficiency = efficiency  # Efficiency factor for PPLNS pools.
        self.consistency_factor = 1.0  # Simulate loyalty benefits.
    
    def calculate_reward(self, row):
        network_hashrate = row['hash_rate'] * 1000
        hashrate_share = self.miner_hashrate / network_hashrate
        blocks_in_period = 18
        block_reward = row['Block reward (BTC)'] + row['transaction fees btc']
        expected_reward = hashrate_share * blocks_in_period * block_reward * self.efficiency * self.consistency_factor
        # Slowly increase the consistency factor, capping at 1.05.
        self.consistency_factor = min(1.05, self.consistency_factor + 0.0001)
        return expected_reward * (1 - self.pool_fee)

class PPSPlusPool(MiningStrategy):
    """
    PPS+ Pool: Pay per share plus additional transaction fees.
    """
    def __init__(self, miner_hashrate, pool_name, fee=0.02, efficiency=1.02):
        super().__init__(f"{pool_name} (PPS+)", miner_hashrate)
        self.pool_fee = fee
        self.efficiency = efficiency
    
    def calculate_reward(self, row):
        network_hashrate = row['hash_rate'] * 1000
        hashrate_share = self.miner_hashrate / network_hashrate
        blocks_in_period = 18
        block_reward = row['Block reward (BTC)'] + row['transaction fees btc']
        expected_reward = hashrate_share * blocks_in_period * block_reward * self.efficiency
        return expected_reward * (1 - self.pool_fee)

class TIDESPool(MiningStrategy):
    """
    TIDES Pool: Time and Difficulty-based Economic Sharing model (e.g., Ocean).
    """
    def __init__(self, miner_hashrate, pool_name, fee=0.0, efficiency=1.0):
        super().__init__(f"{pool_name} (TIDES)", miner_hashrate)
        self.pool_fee = fee
        self.efficiency = efficiency
        self.consistency_factor = 1.0
    
    def calculate_reward(self, row):
        network_hashrate = row['hash_rate'] * 1000
        hashrate_share = self.miner_hashrate / network_hashrate
        blocks_in_period = 18
        block_reward = row['Block reward (BTC)'] + row['transaction fees btc']
        expected_reward = hashrate_share * blocks_in_period * block_reward * self.efficiency * self.consistency_factor
        self.consistency_factor = min(1.05, self.consistency_factor + 0.0001)
        return expected_reward * (1 - self.pool_fee)

class SoloPool(MiningStrategy):
    """
    Solo Pool: Solo mining via a pool interface (e.g., ViaBTC Solo Mode).
    """
    def __init__(self, miner_hashrate, pool_name, fee=0.01):
        super().__init__(f"{pool_name} (SOLO)", miner_hashrate)
        self.pool_fee = fee
        self.blocks_found = 0
    
    def calculate_reward(self, row):
        network_hashrate = row['hash_rate'] * 1000
        p_block = self.miner_hashrate / network_hashrate
        blocks_in_period = 18
        blocks_found = np.random.binomial(blocks_in_period, p_block)
        if blocks_found > 0:
            self.blocks_found += blocks_found
            block_reward = row['Block reward (BTC)'] + row['transaction fees btc']
            return blocks_found * block_reward * (1 - self.pool_fee)
        else:
            return 0

class HybridMining(MiningStrategy):
    """
    Hybrid Mining: Split the miner's hash rate between solo and pool mining.
    """
    def __init__(self, miner_hashrate, solo_ratio=0.5, pool_name="Generic Pool", pool_type="FPPS", pool_fee=0.02):
        super().__init__(f"Hybrid ({int(solo_ratio*100)}% Solo, {int((1-solo_ratio)*100)}% {pool_name})", miner_hashrate)
        self.solo_ratio = solo_ratio
        self.pool_fee = pool_fee
        
        # Split hashrate between solo mining and pool mining.
        self.solo_hashrate = miner_hashrate * solo_ratio
        self.pool_hashrate = miner_hashrate * (1 - solo_ratio)
        
        self.solo_strategy = SoloMining(self.solo_hashrate)
        
        if pool_type.upper() in ["FPPS", "FPPS+"]:
            self.pool_strategy = FPPSPool(self.pool_hashrate, pool_name, fee=pool_fee)
        elif pool_type.upper() == "PPLNS":
            self.pool_strategy = PPLNSPool(self.pool_hashrate, pool_name, fee=pool_fee)
        elif pool_type.upper() == "PPS+":
            self.pool_strategy = PPSPlusPool(self.pool_hashrate, pool_name, fee=pool_fee)
        else:
            self.pool_strategy = FPPSPool(self.pool_hashrate, pool_name, fee=pool_fee)
    
    def calculate_reward(self, row):
        solo_reward = self.solo_strategy.calculate_reward(row)
        pool_reward = self.pool_strategy.calculate_reward(row)
        return solo_reward + pool_reward

class PoolHopping(MiningStrategy):
    """
    Pool Hopping: Dynamically switch between pools based on their current profitability.
    """
    def __init__(self, miner_hashrate, hop_threshold=0.05):
        super().__init__("Pool Hopping", miner_hashrate)
        self.hop_threshold = hop_threshold
        self.hopping_cost = 0.001  # Small cost associated with switching pools
        self.pools = [
            {"name": "Foundry USA", "fee": 0.0, "efficiency": 1.0, "type": "FPPS"},
            {"name": "Antpool", "fee": 0.0, "efficiency": 0.97, "type": "PPLNS"},
            {"name": "Antpool", "fee": 0.04, "efficiency": 1.0, "type": "FPPS"},
            {"name": "ViaBTC", "fee": 0.04, "efficiency": 1.02, "type": "PPS+"},
            {"name": "ViaBTC", "fee": 0.02, "efficiency": 0.97, "type": "PPLNS"},
            {"name": "ViaBTC", "fee": 0.01, "efficiency": 0.95, "type": "SOLO"},
            {"name": "F2Pool", "fee": 0.04, "efficiency": 1.0, "type": "FPPS"},
            {"name": "F2Pool", "fee": 0.02, "efficiency": 0.97, "type": "PPLNS"},
            {"name": "Binance", "fee": 0.04, "efficiency": 1.0, "type": "FPPS"},
            {"name": "SECPOOL", "fee": 0.0, "efficiency": 0.97, "type": "PPLNS"},
            {"name": "SECPOOL", "fee": 0.04, "efficiency": 1.0, "type": "FPPS"},
            {"name": "Spiderpool", "fee": 0.0, "efficiency": 0.97, "type": "PPLNS"},
            {"name": "Spiderpool", "fee": 0.04, "efficiency": 1.0, "type": "FPPS"},
            {"name": "EMCD", "fee": 0.015, "efficiency": 1.02, "type": "FPPS+"},
            {"name": "Luxor", "fee": 0.025, "efficiency": 1.0, "type": "FPPS"},
            {"name": "SBI Crypto", "fee": 0.01, "efficiency": 1.0, "type": "FPPS"},
            {"name": "SBI Crypto", "fee": 0.005, "efficiency": 0.97, "type": "PPLNS"},
            {"name": "NTMinePool", "fee": 0.015, "efficiency": 1.02, "type": "FPPS+"},
            {"name": "Trustpool", "fee": 0.01, "efficiency": 1.02, "type": "PPS+"},
            {"name": "Braiins", "fee": 0.02, "efficiency": 1.0, "type": "FPPS"},
            {"name": "Ocean", "fee": 0.0, "efficiency": 1.0, "type": "TIDES"}
        ]
        self.current_pool = self.pools[0]  
        self.last_hop = None
        self.min_hop_interval = 4  # Minimum intervals before another hop
        self.interval_counter = 0
        self.pool_history = []
    
    def calculate_reward(self, row):
        self.interval_counter += 1
        
        # Evaluate pool options if enough time has passed since the last hop.
        if self.last_hop is None or self.interval_counter >= self.min_hop_interval:
            best_pool = self.current_pool
            best_expected = 0
            
            for pool in self.pools:
                network_hashrate = row['hash_rate'] * 1000
                hashrate_share = self.miner_hashrate / network_hashrate
                blocks_in_period = 18
                block_reward = row['Block reward (BTC)'] + row['transaction fees btc']
                
                if pool["type"] in ["FPPS", "FPPS+"]:
                    actual_efficiency = pool["efficiency"] * (1 + random.uniform(-0.02, 0.02))
                    expected = hashrate_share * blocks_in_period * block_reward * actual_efficiency * (1 - pool["fee"])
                elif pool["type"] == "PPLNS":
                    actual_efficiency = pool["efficiency"] * (1 + random.uniform(-0.03, 0.03))
                    expected = hashrate_share * blocks_in_period * block_reward * actual_efficiency * (1 - pool["fee"])
                elif pool["type"] == "PPS+":
                    actual_efficiency = pool["efficiency"] * (1 + random.uniform(-0.02, 0.02))
                    expected = hashrate_share * blocks_in_period * block_reward * actual_efficiency * (1 - pool["fee"])
                elif pool["type"] == "SOLO":
                    p_block = self.miner_hashrate / network_hashrate
                    expected_blocks = p_block * blocks_in_period
                    expected = expected_blocks * block_reward * (1 - pool["fee"])
                elif pool["type"] == "TIDES":
                    actual_efficiency = pool["efficiency"] * (1 + random.uniform(-0.02, 0.02))
                    expected = hashrate_share * blocks_in_period * block_reward * actual_efficiency * (1 - pool["fee"])
                else:
                    actual_efficiency = pool["efficiency"] * (1 + random.uniform(-0.02, 0.02))
                    expected = hashrate_share * blocks_in_period * block_reward * actual_efficiency * (1 - pool["fee"])
                
                if expected > best_expected:
                    best_expected = expected
                    best_pool = pool
            
            network_hashrate = row['hash_rate'] * 1000
            hashrate_share = self.miner_hashrate / network_hashrate
            blocks_in_period = 18
            block_reward = row['Block reward (BTC)'] + row['transaction fees btc']
            
            if self.current_pool["type"] in ["FPPS", "FPPS+"]:
                current_expected = hashrate_share * blocks_in_period * block_reward * self.current_pool["efficiency"] * (1 - self.current_pool["fee"])
            elif self.current_pool["type"] == "PPLNS":
                current_expected = hashrate_share * blocks_in_period * block_reward * self.current_pool["efficiency"] * (1 - self.current_pool["fee"])
            elif self.current_pool["type"] == "PPS+":
                current_expected = hashrate_share * blocks_in_period * block_reward * self.current_pool["efficiency"] * (1 - self.current_pool["fee"])
            elif self.current_pool["type"] == "SOLO":
                p_block = self.miner_hashrate / network_hashrate
                expected_blocks = p_block * blocks_in_period
                current_expected = expected_blocks * block_reward * (1 - self.current_pool["fee"])
            elif self.current_pool["type"] == "TIDES":
                current_expected = hashrate_share * blocks_in_period * block_reward * self.current_pool["efficiency"] * (1 - self.current_pool["fee"])
            else:
                current_expected = hashrate_share * blocks_in_period * block_reward * self.current_pool["efficiency"] * (1 - self.current_pool["fee"])
            
            # If the best alternative pool offers an improvement above the threshold, hop.
            if (best_expected / current_expected - 1) > self.hop_threshold:
                self.pool_history.append({
                    'timestamp': row.name,
                    'from_pool': f"{self.current_pool['name']} ({self.current_pool['type']})",
                    'to_pool': f"{best_pool['name']} ({best_pool['type']})",
                    'improvement': (best_expected / current_expected - 1) * 100
                })
                self.current_pool = best_pool
                self.last_hop = self.interval_counter
                self.interval_counter = 0
                current_expected -= self.hopping_cost
        
        network_hashrate = row['hash_rate'] * 1000
        hashrate_share = self.miner_hashrate / network_hashrate
        blocks_in_period = 18
        block_reward = row['Block reward (BTC)'] + row['transaction fees btc']
        
        if self.current_pool["type"] in ["FPPS", "FPPS+"]:
            expected_reward = hashrate_share * blocks_in_period * block_reward * self.current_pool["efficiency"]
        elif self.current_pool["type"] == "PPLNS":
            expected_reward = hashrate_share * blocks_in_period * block_reward * self.current_pool["efficiency"]
        elif self.current_pool["type"] == "PPS+":
            expected_reward = hashrate_share * blocks_in_period * block_reward * self.current_pool["efficiency"]
        elif self.current_pool["type"] == "SOLO":
            p_block = self.miner_hashrate / network_hashrate
            blocks_found = np.random.binomial(blocks_in_period, p_block)
            if blocks_found > 0:
                return blocks_found * block_reward * (1 - self.current_pool["fee"])
            else:
                return 0
        elif self.current_pool["type"] == "TIDES":
            expected_reward = hashrate_share * blocks_in_period * block_reward * self.current_pool["efficiency"]
        else:
            expected_reward = hashrate_share * blocks_in_period * block_reward * self.current_pool["efficiency"]
        
        return expected_reward * (1 - self.current_pool["fee"])

# ============================================================================
# NEW STRATEGIES
# ============================================================================

class GreenMining(MiningStrategy):
    """
    Energy Cost-Optimized (Green) Mining Strategy:
    Adjust operations based on a specified energy cost threshold.
    Assumes each row has an "energy_cost" column (USD per period) and uses 
    the "BTC market price usd" for conversion.
    """
    def __init__(self, miner_hashrate, pool_name="Green Pool", fee=0.02, efficiency=1.0, energy_threshold=1000):
        super().__init__(f"{pool_name} (Green Mining)", miner_hashrate)
        self.pool_fee = fee
        self.efficiency = efficiency
        self.energy_threshold = energy_threshold  # USD threshold per period
    
    def calculate_reward(self, row):
        network_hashrate = row['hash_rate'] * 1000
        hashrate_share = self.miner_hashrate / network_hashrate
        blocks_in_period = 18
        block_reward = row['Block reward (BTC)'] + row['transaction fees btc']
        base_reward = hashrate_share * blocks_in_period * block_reward * self.efficiency * (1 - self.pool_fee)
        
        # Retrieve energy cost (USD) for this period; assume 0 if not present.
        energy_cost_usd = row.get("energy_cost", 0)
        btc_price = row["BTC market price usd"]
        energy_cost_btc = energy_cost_usd / btc_price if btc_price else 0
        
        # If the energy cost is above the threshold, the miner shuts down for the period.
        operational_factor = 1 if energy_cost_usd <= self.energy_threshold else 0
        
        net_reward = base_reward * operational_factor - energy_cost_btc
        return max(net_reward, 0)

class MultiPoolMining(MiningStrategy):
    """
    Multi-Pool (Diversified) Mining Strategy:
    Split the miner's hash rate across multiple pool strategies as defined by weightings.
    The overall reward is the sum of the rewards from each sub-strategy.
    
    pool_allocations: List of dictionaries, each with keys:
        - "type": Pool type (e.g., "FPPS", "PPLNS", "PPS+")
        - "pool_name": Name of the pool
        - "allocation": Fraction of total hash rate allocated (should sum to 1)
        - Optionally, "fee" and "efficiency".
    """
    def __init__(self, miner_hashrate, pool_allocations):
        super().__init__("Multi-Pool Mining", miner_hashrate)
        self.pool_allocations = pool_allocations
        self.sub_strategies = []
        for alloc in self.pool_allocations:
            allocated_hashrate = miner_hashrate * alloc.get("allocation", 0)
            fee = alloc.get("fee", 0.02)
            efficiency = alloc.get("efficiency", 1.0)
            pool_name = alloc.get("pool_name", "Generic Pool")
            pool_type = alloc.get("type", "FPPS")
            
            if pool_type.upper() in ["FPPS", "FPPS+"]:
                strat = FPPSPool(allocated_hashrate, pool_name, fee=fee, efficiency=efficiency)
            elif pool_type.upper() == "PPLNS":
                strat = PPLNSPool(allocated_hashrate, pool_name, fee=fee, efficiency=efficiency)
            elif pool_type.upper() == "SOLO":
                strat = SoloPool(allocated_hashrate, pool_name, fee=fee)
            else:
                # Default to FPPS pooling.
                strat = FPPSPool(allocated_hashrate, pool_name, fee=fee, efficiency=efficiency)
            
            self.sub_strategies.append(strat)
    
    def calculate_reward(self, row):
        total_reward = 0
        for strat in self.sub_strategies:
            total_reward += strat.calculate_reward(row)
        return total_reward

# ============================================================================
# STRATEGY CREATION & SIMULATION
# ============================================================================

def create_mining_pool_strategies(miner_hashrate):
    """
    Create a list of mining strategies (both original and new) that will be simulated.
    """
    pool_strategies = [
        # Original strategies:
        SoloMining(miner_hashrate),
        FPPSPool(miner_hashrate, "Foundry USA", fee=0.0),
        PPLNSPool(miner_hashrate, "Antpool", fee=0.0, efficiency=0.97),
        FPPSPool(miner_hashrate, "Antpool", fee=0.04),
        PPSPlusPool(miner_hashrate, "ViaBTC", fee=0.04, efficiency=1.02),
        PPLNSPool(miner_hashrate, "ViaBTC", fee=0.02, efficiency=0.97),
        SoloPool(miner_hashrate, "ViaBTC", fee=0.01),
        FPPSPool(miner_hashrate, "F2Pool", fee=0.04),
        PPLNSPool(miner_hashrate, "F2Pool", fee=0.02, efficiency=0.97),
        FPPSPool(miner_hashrate, "Binance", fee=0.04),
        PPLNSPool(miner_hashrate, "SECPOOL", fee=0.0, efficiency=0.97),
        FPPSPool(miner_hashrate, "SECPOOL", fee=0.04),
        PPLNSPool(miner_hashrate, "Spiderpool", fee=0.0, efficiency=0.97),
        PPSPlusPool(miner_hashrate, "EMCD", fee=0.015, efficiency=1.02),
        FPPSPool(miner_hashrate, "Luxor", fee=0.025),
        FPPSPool(miner_hashrate, "SBI Crypto", fee=0.01),
        PPLNSPool(miner_hashrate, "SBI Crypto", fee=0.005, efficiency=0.97),
        TIDESPool(miner_hashrate, "Ocean", fee=0.0),
        HybridMining(miner_hashrate, solo_ratio=0.3, pool_name="Foundry USA", pool_type="FPPS", pool_fee=0.0),
        HybridMining(miner_hashrate, solo_ratio=0.3, pool_name="ViaBTC", pool_type="PPLNS", pool_fee=0.02),
        PoolHopping(miner_hashrate, hop_threshold=0.05),
        # New strategies:
        GreenMining(miner_hashrate, pool_name="EcoPool", fee=0.02, efficiency=1.0, energy_threshold=1000),
        MultiPoolMining(miner_hashrate, pool_allocations=[
            {"type": "FPPS", "pool_name": "Foundry USA", "allocation": 0.4, "fee": 0.0, "efficiency": 1.0},
            {"type": "PPLNS", "pool_name": "Antpool", "allocation": 0.3, "fee": 0.0, "efficiency": 0.97},
            {"type": "PPS+", "pool_name": "ViaBTC", "allocation": 0.3, "fee": 0.04, "efficiency": 1.02}
        ])
    ]
    return pool_strategies

def run_mining_simulation(file_path, miner_size="Medium"):
    """
    Run the mining simulation for the given miner size.
    
    miner_size options: "Small", "Medium", "Large", "Industrial"
    Each is mapped to a specific miner hashrate (TH/s).
    """
    miner_sizes = {
        "Small": 10000,            # 10 PH/s
        "Medium": 100000,          # 100 PH/s
        "Large": 1000000,          # 1 EH/s
        "Industrial": 10000000     # 10 EH/s
    }
    miner_hashrate = miner_sizes.get(miner_size, 10000)
    print(f"Running simulation for {miner_size} miner with {miner_hashrate} TH/s")
    
    df = load_data(file_path)
    
    avg_network_hashrate = df['hash_rate'].mean() * 1000  # in TH/s
    for size, hashrate in miner_sizes.items():
        network_share = hashrate / avg_network_hashrate
        blocks_in_period = 18
        prob_per_period = 1 - (1 - network_share) ** blocks_in_period
        expected_blocks = prob_per_period * len(df)
        print(f"{size} miner: {network_share*100:.6f}% of network, {prob_per_period*100:.6f}% chance per period, expected {expected_blocks:.2f} blocks over dataset")
    
    strategies = create_mining_pool_strategies(miner_hashrate)
    
    results = {}
    for strategy in strategies:
        print(f"Simulating {strategy.name}...")
        result_df = strategy.run_simulation(df)
        results[strategy.name] = result_df
        
        if isinstance(strategy, SoloMining) or isinstance(strategy, SoloPool):
            blocks_found = getattr(strategy, 'blocks_found', 0)
            print(f"{strategy.name} blocks found: {blocks_found}")
            if blocks_found > 0 and hasattr(strategy, 'block_timestamps'):
                print("Block find details:")
                for i, (timestamp, reward) in enumerate(zip(strategy.block_timestamps, strategy.block_rewards)):
                    print(f"  Block {i+1}: {timestamp} - {reward:.8f} BTC")
            network_share = miner_hashrate / avg_network_hashrate
            blocks_in_period = 18
            prob_per_period = 1 - (1 - network_share) ** blocks_in_period
            expected_blocks = prob_per_period * len(df)
            print(f"Expected blocks: {expected_blocks:.2f}, Actual blocks: {blocks_found}")
            print(f"Ratio: {blocks_found / expected_blocks if expected_blocks > 0 else 0:.2f}")
        
        if isinstance(strategy, PoolHopping) and hasattr(strategy, 'pool_history'):
            print(f"Pool hopping transitions: {len(strategy.pool_history)}")
            if len(strategy.pool_history) > 0:
                print("Top 5 pool transitions:")
                for i, hop in enumerate(strategy.pool_history[:5]):
                    print(f"  {i+1}: {hop['from_pool']} -> {hop['to_pool']} ({hop['improvement']:.2f}% improvement)")
    
    # Combine all strategy results into one dataframe for further analysis.
    combined_results = pd.DataFrame(index=df.index)
    
    for strategy_name, result_df in results.items():
        combined_results[f"{strategy_name}_reward"] = result_df["reward"]
        combined_results[f"{strategy_name}_cumulative"] = result_df["cumulative_reward"]
    
    # Convert cumulative BTC rewards to USD using BTC market price.
    for strategy_name in [s.name for s in strategies]:
        btc_cumulative = combined_results[f"{strategy_name}_cumulative"]
        combined_results[f"{strategy_name}_cumulative_usd"] = btc_cumulative * df["BTC market price usd"]
    
    return combined_results, df, strategies

# ============================================================================
# ANALYSIS & VISUALIZATION FUNCTIONS
# ============================================================================

def analyze_results(combined_results, original_df, strategies, miner_size):
    """
    Analyze and visualize simulation results for a given miner size.
    """
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # 1. Plot cumulative BTC rewards over time for top 5 strategies.
    strategy_totals = [(s.name, combined_results[f"{s.name}_cumulative"].iloc[-1]) for s in strategies]
    top_strategies = sorted(strategy_totals, key=lambda x: x[1], reverse=True)[:5]
    
    plt.figure(figsize=(14, 8))
    for strategy_name, _ in top_strategies:
        plt.plot(combined_results.index, combined_results[f"{strategy_name}_cumulative"], label=strategy_name)
    
    plt.title(f"Cumulative BTC Rewards - Top 5 Strategies ({miner_size} Miner)")
    plt.xlabel("Date")
    plt.ylabel("BTC")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/{miner_size}_top5_cumulative_btc.png")
    plt.close()
    
    # 2. Plot cumulative USD value over time for top 5 strategies.
    plt.figure(figsize=(14, 8))
    for strategy_name, _ in top_strategies:
        plt.plot(combined_results.index, combined_results[f"{strategy_name}_cumulative_usd"], label=strategy_name)
    
    plt.title(f"Cumulative USD Value - Top 5 Strategies ({miner_size} Miner)")
    plt.xlabel("Date")
    plt.ylabel("USD")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/{miner_size}_top5_cumulative_usd.png")
    plt.close()
    
    # 3. Plot reward volatility (rolling standard deviation).
    window_size = 30  # For example, 30 time periods.
    plt.figure(figsize=(14, 8))
    for strategy_name, _ in top_strategies:
        rolling_std = combined_results[f"{strategy_name}_reward"].rolling(window=window_size).std()
        plt.plot(combined_results.index, rolling_std, label=strategy_name)
    
    plt.title(f"30-Period Rolling Volatility - Top 5 Strategies ({miner_size} Miner)")
    plt.xlabel("Date")
    plt.ylabel("BTC (Std. Dev.)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/{miner_size}_top5_volatility.png")
    plt.close()
    
    # 4. Calculate monthly returns for each strategy.
    monthly_returns = {}
    for strategy in strategies:
        monthly_rewards = combined_results[f"{strategy.name}_reward"].resample('ME').sum()
        monthly_returns[strategy.name] = monthly_rewards
    
    monthly_df = pd.DataFrame(monthly_returns)
    
    plt.figure(figsize=(14, 8))
    monthly_df[[name for name, _ in top_strategies]].plot(kind='bar', figsize=(14, 8))
    plt.title(f"Monthly BTC Rewards - Top 5 Strategies ({miner_size} Miner)")
    plt.xlabel("Month")
    plt.ylabel("BTC")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/{miner_size}_top5_monthly_rewards.png")
    plt.close()
    
    # 5. Calculate risk-adjusted returns (using a Sharpe ratio-like measure).
    risk_adjusted = {}
    for strategy in strategies:
        monthly_reward_series = monthly_df[strategy.name]
        mean_return = monthly_reward_series.mean()
        std_return = monthly_reward_series.std()
        if mean_return > 0 and std_return > 0:
            risk_adjusted[strategy.name] = mean_return / std_return
        else:
            risk_adjusted[strategy.name] = 0
    
    # 6. Summary statistics.
    summary = {
        "Total BTC Mined": {s.name: combined_results[f"{s.name}_cumulative"].iloc[-1] for s in strategies},
        "Final USD Value": {s.name: combined_results[f"{s.name}_cumulative_usd"].iloc[-1] for s in strategies},
        "Risk-Adjusted Return": risk_adjusted,
        "Avg Monthly Return (BTC)": {s.name: monthly_df[s.name].mean() for s in strategies},
        "Return Volatility (σ)": {s.name: monthly_df[s.name].std() for s in strategies},
    }
    
    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values(by="Total BTC Mined", ascending=False)
    
    return {"summary": summary_df, "monthly_returns": monthly_df, "combined_results": combined_results}

def compare_miner_sizes(all_results):
    """
    Compare simulation results across different miner sizes.
    """
    top_strategies = {}
    for miner_size, analysis in all_results.items():
        top5 = analysis["summary"].head(5).index.tolist()
        top_strategies[miner_size] = top5
    
    print("\nTop 5 strategies by miner size:")
    for size, strategies in top_strategies.items():
        print(f"{size}: {', '.join(strategies)}")
    
    # Compile total BTC mined by strategy across miner sizes.
    all_strategies = set()
    for strategies in top_strategies.values():
        all_strategies.update(strategies)
    
    btc_by_strategy_size = {}
    for strategy in all_strategies:
        btc_values = {}
        for miner_size, analysis in all_results.items():
            if strategy in analysis["summary"].index:
                btc_values[miner_size] = analysis["summary"].loc[strategy, "Total BTC Mined"]
        btc_by_strategy_size[strategy] = btc_values
    
    btc_df = pd.DataFrame(btc_by_strategy_size).T
    
    plt.figure(figsize=(16, 10))
    btc_df.plot(kind='bar', figsize=(16, 10))
    plt.title("Total BTC Mined by Strategy and Miner Size")
    plt.xlabel("Strategy")
    plt.ylabel("Total BTC")
    plt.legend(title="Miner Size")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("plots/strategy_comparison_by_size.png")
    plt.close()
    
    # Rank strategies by total BTC mined.
    rankings = {}
    for miner_size, analysis in all_results.items():
        ranked = analysis["summary"]["Total BTC Mined"].rank(ascending=False)
        rankings[miner_size] = ranked
    
    rank_df = pd.DataFrame(rankings)
    rank_df['avg_rank'] = rank_df.mean(axis=1)
    best_overall = rank_df.sort_values('avg_rank').head(5)
    
    print("\nBest Overall Strategies (Average Rank Across Miner Sizes):")
    for strategy, avg_rank in best_overall['avg_rank'].items():
        print(f"{strategy}: {avg_rank:.2f}")
    
    risk_adjusted = {}
    for strategy in best_overall.index:
        risk_values = {}
        for miner_size, analysis in all_results.items():
            if strategy in analysis["summary"].index:
                risk_values[miner_size] = analysis["summary"].loc[strategy, "Risk-Adjusted Return"]
        risk_adjusted[strategy] = risk_values
    
    risk_df = pd.DataFrame(risk_adjusted).T
    
    plt.figure(figsize=(14, 8))
    risk_df.plot(kind='bar', figsize=(14, 8))
    plt.title("Risk-Adjusted Returns for Top Strategies")
    plt.xlabel("Strategy")
    plt.ylabel("Risk-Adjusted Return")
    plt.legend(title="Miner Size")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/risk_adjusted_top_strategies.png")
    plt.close()
    
    return {"top_strategies": top_strategies, "strategy_comparison": btc_df, "strategy_rankings": rank_df}

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    file_path = 'bitcoin_metrics_and_energy_updated_final_manually_scaled.xlsx'
    miner_sizes = ["Small", "Medium", "Large", "Industrial"]
    
    all_results = {}
    
    for miner_size in miner_sizes:
        print(f"Running analysis for {miner_size} miner")
        
        combined_results, original_df, strategies = run_mining_simulation(file_path, miner_size)
        analysis = analyze_results(combined_results, original_df, strategies, miner_size)
        
        print(f"\nSummary for {miner_size} miner (Top 10 strategies):")
        print(analysis["summary"].head(10))
        
        all_results[miner_size] = analysis
    
    comparison = compare_miner_sizes(all_results)
    
    return all_results

if __name__ == "__main__":
    all_results = main()

