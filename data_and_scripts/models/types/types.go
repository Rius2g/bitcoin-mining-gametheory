package types

import (
    "math/big"
)

type MinerType struct {
    Name string 
    MinHash float64 
    MaxHash float64 
}

type Miner struct {
    PowerEfficiency float64
    IndividualHashrate big.Int 
    MinerType MinerType
}

var MinerTypes = []MinerType{
    {"Small", 90, 90},
    {"Medium", 180, 450},
    {"Large", 540, 900},
    {"Industrial", 990, 900000},
}

type Strategy int

const (
    SoloMining Strategy = iota 
    PoolMining 
    NoMining
    SwitchingPool
)

type MiningPool struct {
    Name string
    FeePPLNS float64
    FeeFPPS float64 
    FeePPSPlus float64
    NetworkShare float64
    PayoutMethod string
}

// Extended SimulationResult with date range information
type SimulationResult struct {
    MinerType string 
    Hashrate float64 
    Strategy Strategy 
    TimeInterval float64 
    Region string
    NetProfit float64 
    ROI float64
    StartDate string // Start date of the simulation period
    EndDate string   // End date of the simulation period
}

type Region struct {
    Name string
    CostField string
}

// DataRow represents a single row from the Excel file
// Field names match the Excel column headers we observed
type DataRow struct {
    Timestamp string              // timestamp
    TotalHashrate float64         // hash_rate
    ConvertedHash float64         // Converted hash
    BTCPriceUSD float64           // BTC market price usd
    TransactionFees float64       // transaction fees btc
    TransactionFeesUSD float64    // Average transaction fees usd (3-hour interval)
    ExchangeRate float64          // Exchange rate (EUR/USD)
    DynamicExchangeRate float64   // Dynnamic exchange rate Denmark price usd kwh
    DenmarkPriceKWh float64       // denmark price usd kwh
    TexasPriceKWh float64         // texas price usd kwh
    KazakhstanPriceKWh float64    // kazakhstan price usd kwh
    ChinaPriceKWh float64         // china price usd kwh
    BlockReward float64           // Block reward (BTC)
}
