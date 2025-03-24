package payout
import (
    "math/big"
    "math/rand"
    "time"
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
func CalculatePayout(
    BTCprice float64,
    totalHashrate *big.Int,
    miner t.Miner,
    electricityCost float64,
    transactionFees float64,
    joinedPool bool,
    pool t.MiningPool,
    blockReward float64,
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
        // Solo mining - simulate the random chance of finding blocks
        // Initialize random number generator with current time
        source := rand.NewSource(time.Now().UnixNano())
        r := rand.New(source)
        
        // Simulate each block in the time period
        btcEarned := 0.0
        for range(int(totalBlocks)) {
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
    
    // 5) Decide whether to mine
    if expectedRevenueUSD <= totalElectricityCost {
        // Mining is unprofitable - don't mine
        return 0.0
    }
    
    // 6) Net profit
    netProfit := expectedRevenueUSD - totalElectricityCost
    
    return netProfit
}
