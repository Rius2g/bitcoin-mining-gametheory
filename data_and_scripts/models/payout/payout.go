package payout
import (
    "math/big"
    "math/rand"
    t "models/types"
)

// calculatePayout estimates the net profit (USD) the miner earns
// for one block event (or per-block expected payout) given:
//  - BTCprice: USD price of Bitcoin
//  - totalHashrate: total network hash in H/s (as *big.Int)
//  - miner: holds the miner's hash rate (big.Int) & hardware info
//  - electricityCost: electricity cost in USD per kWh
//  - transactionFees: total block transaction fees in BTC
//  - joinedPool: whether miner is in a pool
//  - pool: if joinedPool=true, specify which pool + fee method
//  - blockReward: e.g. 6.25 BTC
//  - r: random number generator (passed in from simulation to avoid creating one per call)
func CalculatePayout(
    BTCprice float64,
    totalHashrate *big.Int,
    miner t.Miner,
    electricityCost float64,
    transactionFees float64,
    joinedPool bool,
    pool t.MiningPool,
    blockReward float64,
    r *rand.Rand,
) float64 {
    // 1) Convert hashrates to floating-point
    minerHashF := new(big.Float).SetInt(&miner.IndividualHashrate)
    totalHashF := new(big.Float).SetInt(totalHashrate)
    // Convert to TH/s for efficiency calculations
    hashTHFloat := new(big.Float).Quo(minerHashF, big.NewFloat(1e12))
    hashTH, _ := hashTHFloat.Float64()
    
    // 2) Calculate electricity cost first to decide whether to mine
    powerWatts := hashTH * miner.PowerEfficiency
    kWhUsed := (powerWatts / 1000.0) * 3 
    totalElectricityCost := kWhUsed * electricityCost
    
    // 3) Calculate miner's share of network hashrate (probability)
    ratio := new(big.Float).Quo(minerHashF, totalHashF)
    relativeHashRate, _ := ratio.Float64()
    
    // Total block reward in BTC
    totalRewardBTC := blockReward + transactionFees
    // Total blocks in the time period (3hour interval (6 blocks/hour) * 3)
    totalBlocks := 18.0
    
    var expectedRevenueUSD float64
    
    // 4) Different calculation based on solo vs pool mining
    if joinedPool {
        // Pool mining - predictable payout based on hashrate contribution
        expectedBTC := relativeHashRate * totalRewardBTC * totalBlocks
        
        // Apply pool fees
        var feeRate float64
        switch pool.PayoutMethod {
        case "PPLNS":
            feeRate = pool.FeePPLNS
        case "FPPS":
            feeRate = pool.FeeFPPS
        case "PPS+":
            feeRate = pool.FeePPSPlus
        default:
            feeRate = 0.0
        }
        expectedBTC *= (1.0 - feeRate)
        
        // Convert to USD
        expectedRevenueUSD = expectedBTC * BTCprice
    } else {
        // Solo mining - use probability distribution instead of per-block simulation
        // For low probability events, we can use binomial distribution
        
        // Calculate expected number of blocks found using binomial distribution
        // For solo mining with extremely low hashrates, most simulations will yield 0 blocks
        // We can optimize by sampling from the binomial distribution directly
        if relativeHashRate < 0.001 { // For very small miners (less than 0.1% of network)
            // For small probabilities, we can approximate with Poisson distribution
            // or use a shortcut: directly sample if we find any blocks
            
            // Probability of finding at least one block
            probAtLeastOne := 1.0 - pow(1.0-relativeHashRate, totalBlocks)
            
            // Generate a random number to determine if any blocks are found
            if r.Float64() <= probAtLeastOne {
                // Found at least one block
                // For very low probabilities, it's most likely just one block
                // but we could sample from geometric distribution for more accuracy
                btcEarned := totalRewardBTC // Assume one block for simplicity
                expectedRevenueUSD = btcEarned * BTCprice
            } else {
                // No blocks found
                expectedRevenueUSD = 0
            }
        } else {
            // For larger miners, simulate each block
            // This is faster than the original approach because we're using a passed-in RNG
            btcEarned := 0.0
            for i := 0; i < int(totalBlocks); i++ {
                // Generate random number between 0 and 1
                chance := r.Float64()
                
                // If random number falls within the miner's hashrate percentage,
                // they found the block
                if chance <= relativeHashRate {
                    btcEarned += totalRewardBTC
                }
            }
            
            // Convert to USD
            expectedRevenueUSD = btcEarned * BTCprice
        }
    }
    
    // 5) Decide whether to mine
    if expectedRevenueUSD <= totalElectricityCost {
        // Mining is unprofitable - don't mine
        return 0.0
    }
    
    // 6) Net profit
    netProfit := expectedRevenueUSD - totalElectricityCost
    
    return netProfit
}

// Helper function to calculate power
func pow(base, exponent float64) float64 {
    result := 1.0
    for i := 0; i < int(exponent); i++ {
        result *= base
    }
    return result
}
