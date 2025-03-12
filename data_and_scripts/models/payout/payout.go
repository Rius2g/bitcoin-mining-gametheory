package payout

import (
    "math/big"
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
//  - miningDurationHours: how many hours the miner runs
func CalculatePayout(
    BTCprice float64,             // e.g. 27000.0
    totalHashrate *big.Int,       
    miner t.Miner,                
    electricityCost float64,      // e.g. $0.05 per kWh
    transactionFees float64,      // e.g. 0.75 BTC
    joinedPool bool,
    pool t.MiningPool,
    blockReward float64,          // e.g. 6.25 BTC
    miningDurationHours float64,  // e.g. 24 (one day)
) float64 {

    // 1) Convert miner's & network's hashrate to floating-point
    minerHashF := new(big.Float).SetInt(&miner.IndividualHashrate)
    totalHashF := new(big.Float).SetInt(totalHashrate)

    // 2) ratio = (miner’s hash / total network hash)
    ratio := new(big.Float).Quo(minerHashF, totalHashF)  // BigFloat ratio
    relativeHashRate, _ := ratio.Float64()               // Convert to float64

    // 3) Compute block reward + fees in BTC
    totalRewardBTC := blockReward + transactionFees

    // 4) Miner’s expected BTC from this block:
    //    share = ratio * (block reward + tx fees)
    minerRewardBTC := relativeHashRate * totalRewardBTC

    // 5) Apply pool fees if joined
    if joinedPool {
        // Select fee based on pool's payout method (PPLNS, PPS+, FPPS, etc.)
        var feeRate float64
        switch pool.PayoutMethod {
        case "PPLNS":
            feeRate = pool.FeePPLNS
        case "FPPS":
            feeRate = pool.FeeFPPS
        case "PPS+":
            feeRate = pool.FeePPSPlus
        default:
            feeRate = 0.0 // fallback or handle error
        }
        // Deduct the fee
        minerRewardBTC *= (1.0 - feeRate)
    }

    // 6) Convert final miner BTC to USD
    rewardUSD := minerRewardBTC * BTCprice

    // 7) Calculate electricity usage cost
    //    Assuming miner.PowerEfficiency is in W/TH (or W per unit of hashrate)
    //    We have to ensure consistent units:
    //    (a) If miner.IndividualHashrate is in H/s, we need TH/s.
    //    (b) If it’s already TH/s, skip conversion.

    // Example: If the big.Int is in H/s, convert to TH/s by dividing by 1e12
    // (You might already store TH/s as an integer, adapt as needed.)
    hashTHFloat := new(big.Float).Quo(minerHashF, big.NewFloat(1e12))
    hashTH, _ := hashTHFloat.Float64() 

    // Now, total power usage (W) = TH * (W/TH)
    powerWatts := hashTH * miner.PowerEfficiency

    // Convert W to kW => divide by 1000
    // Then multiply by hours for total kWh
    kWhUsed := (powerWatts / 1000.0) * miningDurationHours

    // Electricity cost in USD
    totalElectricityCost := kWhUsed * electricityCost

    // 8) Net profit
    netProfit := rewardUSD - totalElectricityCost
    return netProfit
}

