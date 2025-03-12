package main

import (
    t "models/types"
    payout "models/payout"
    dataloader "models/dataloader"
    "log"
    "fmt"
    "time"
    "math/rand"
    "math/big"
    "sort"
)

// Time intervals in hours
const (
    DailyInterval   = 24
    WeeklyInterval  = 24 * 7
    MonthlyInterval = 24 * 30
    YearlyInterval  = 24 * 365
)

// Current block reward (BTC)
const blockReward = 6.25

// Define mining pools with their fees and network share
var miningPools = []t.MiningPool{
    {Name: "Binance pool", FeePPLNS: 0.0, FeeFPPS: 0.04, FeePPSPlus: 0.0, NetworkShare: 0.073, PayoutMethod: "FPPS"},
    {Name: "Antpool", FeePPLNS: 0.0, FeeFPPS: 0.0, FeePPSPlus: 0.04, NetworkShare: 0.169, PayoutMethod: "PPLNS"},
    {Name: "F2Pool", FeePPLNS: 0.02, FeeFPPS: 0.04, FeePPSPlus: 0.0, NetworkShare: 0.1055, PayoutMethod: "PPLNS"},
    {Name: "BTC.com", FeePPLNS: 0.02, FeeFPPS: 0.0, FeePPSPlus: 0.04, NetworkShare: 0.128, PayoutMethod: "PPS+"},
    {Name: "Foundry USA", FeePPLNS: 0.0, FeeFPPS: 0.0, FeePPSPlus: 0.0, NetworkShare: 0.29, PayoutMethod: "FPPS"},
    {Name: "Secpool", FeePPLNS: 0.0, FeeFPPS: 0.04, FeePPSPlus: 0.0, NetworkShare: 0.036, PayoutMethod: "PPLNS"},
    {Name: "Spiderpool", FeePPLNS: 0.0, FeeFPPS: 0.04, FeePPSPlus: 0.0, NetworkShare: 0.036, PayoutMethod: "PPLNS"},
    {Name: "Luxor", FeePPLNS: 0.0, FeeFPPS: 0.025, FeePPSPlus: 0.0, NetworkShare: 0.036, PayoutMethod: "FPPS"},
}

// Define regions with their power costs
var regions = []t.Region{
    {Name: "China", CostField: "ChinaPriceKWh"},
    {Name: "Kazakhstan", CostField: "KazakhstanPriceKWh"},
    {Name: "Texas", CostField: "TexasPriceKWh"},
    {Name: "Denmark", CostField: "DenmarkPriceKWh"},
}

// Function to create a random miner
func createRandomMiner(r *rand.Rand) t.Miner {
    // Select random miner type
    minerTypeIdx := r.Intn(len(t.MinerTypes))
    minerType := t.MinerTypes[minerTypeIdx]
    
    // Generate random hashrate within the miner type's range
    hashDiff := minerType.MaxHash - minerType.MinHash
    hashrate := minerType.MinHash
    if hashDiff > 0 {
        hashrate += r.Float64() * hashDiff
    }
    
    // Convert hashrate to TH/s and then to H/s for big.Int representation
    hashrateHs := new(big.Int).SetInt64(int64(hashrate * 1e12))
    
    // Set a reasonable power efficiency (W/TH) based on miner type
    // Modern ASIC efficiency ranges from 20-40 W/TH
    efficiencyWpTh := 20.0 + r.Float64()*20.0
    
    return t.Miner{
        PowerEfficiency:   efficiencyWpTh,
        IndividualHashrate: *hashrateHs,
        MinerType:         minerType,
    }
}

// Function to select random time interval
func getRandomTimeInterval(r *rand.Rand) float64 {
    intervals := []float64{DailyInterval, WeeklyInterval, MonthlyInterval, YearlyInterval}
    return intervals[r.Intn(len(intervals))]
}

// Function to select random strategy
func getRandomStrategy(r *rand.Rand) t.Strategy {
    return t.Strategy(r.Intn(4))
}

// Function to select random pool
func getRandomPool(r *rand.Rand) t.MiningPool {
    return miningPools[r.Intn(len(miningPools))]
}

// Function to select random region
func getRandomRegion(r *rand.Rand) t.Region {
    return regions[r.Intn(len(regions))]
}

// Function to get electricity cost based on region and data row
func getElectricityCost(region t.Region, dataRow t.DataRow) float64 {
    switch region.CostField {
    case "ChinaPriceKWh":
        return dataRow.ChinaPriceKWh
    case "KazakhstanPriceKWh":
        return dataRow.KazakhstanPriceKWh
    case "TexasPriceKWh":
        return dataRow.TexasPriceKWh
    case "DenmarkPriceKWh":
        return dataRow.DenmarkPriceKWh
    default:
        return 0.1 // Default value if not found
    }
}

// Get hours per time interval (for selecting proper chunks of data)
func getHoursFromTimeInterval(interval float64) int {
    switch interval {
    case DailyInterval:
        return 24
    case WeeklyInterval:
        return 24 * 7
    case MonthlyInterval:
        return 24 * 30
    case YearlyInterval:
        return 24 * 365
    default:
        return 24 // Default to daily
    }
}

// Convert hours to number of 3-hour periods (data rows)
func hoursToDataPeriods(hours int) int {
    return hours / 3 // Each data row represents a 3-hour period
}

// Function to run a single simulation with continuous time intervals
func runSimulation(dataRows []t.DataRow, r *rand.Rand) t.SimulationResult {
    // 1. Create random miner
    miner := createRandomMiner(r)
    
    // 2. Select random time interval
    timeInterval := getRandomTimeInterval(r)
    
    // 3. Select random strategy
    strategy := getRandomStrategy(r)
    
    // 4. Select random region for electricity costs
    region := getRandomRegion(r)
    
    // 5. Determine how many consecutive data rows we need
    hoursNeeded := getHoursFromTimeInterval(timeInterval)
    periodsNeeded := hoursToDataPeriods(hoursNeeded)
    
    // Make sure we don't exceed the available data
    if periodsNeeded >= len(dataRows) {
        periodsNeeded = len(dataRows) / 2 // Limit to half the available data
    }
    
    // 6. Select a random starting point in the data, ensuring we have enough data ahead
    maxStartIndex := max(0, len(dataRows) - periodsNeeded -1) 
    startIdx := r.Intn(maxStartIndex)
    
    // 7. Initialize variables for profit calculation
    var totalProfit float64 = 0
    
    // 8. Execute strategy over the continuous time interval
    switch strategy {
    case t.SoloMining:
        // Solo mining over the entire period
        for i := range periodsNeeded {
            dataRow := dataRows[startIdx+i]
            electricityCost := getElectricityCost(region, dataRow)
            totalHashrate := new(big.Int).SetInt64(int64(dataRow.TotalHashrate * 1e12))
            
            // Calculate profit for this 3-hour period
            periodProfit := payout.CalculatePayout(
                dataRow.BTCPriceUSD,
                totalHashrate,
                miner,
                electricityCost,
                dataRow.TransactionFees,
                false, // Not in a pool
                t.MiningPool{}, // Empty pool (not used)
                blockReward,
                3, // Each data row is a 3-hour period
            )
            
            totalProfit += periodProfit
        }
    
    case t.PoolMining:
        // Select a random pool for the whole period
        pool := getRandomPool(r)
        
        for i := range periodsNeeded {
            dataRow := dataRows[startIdx+i]
            electricityCost := getElectricityCost(region, dataRow)
            totalHashrate := new(big.Int).SetInt64(int64(dataRow.TotalHashrate * 1e12))
            
            // Calculate profit for this 3-hour period
            periodProfit := payout.CalculatePayout(
                dataRow.BTCPriceUSD,
                totalHashrate,
                miner,
                electricityCost,
                dataRow.TransactionFees,
                true, // In a pool
                pool,
                blockReward,
                3, // Each data row is a 3-hour period
            )
            
            totalProfit += periodProfit
        }
    
    case t.NoMining:
        // No mining means no profit, but also no electricity cost
        totalProfit = 0
    
    case t.SwitchingPool:
        // Switch pools halfway through the time interval
        pool1 := getRandomPool(r)
        pool2 := getRandomPool(r)
        
        switchPoint := periodsNeeded / 2
        
        for i := range periodsNeeded {
            dataRow := dataRows[startIdx+i]
            electricityCost := getElectricityCost(region, dataRow)
            totalHashrate := new(big.Int).SetInt64(int64(dataRow.TotalHashrate * 1e12))
            
            // Use the appropriate pool based on where we are in the time interval
            currentPool := pool1
            if i >= switchPoint {
                currentPool = pool2
            }
            
            // Calculate profit for this 3-hour period
            periodProfit := payout.CalculatePayout(
                dataRow.BTCPriceUSD,
                totalHashrate,
                miner,
                electricityCost,
                dataRow.TransactionFees,
                true, // In a pool
                currentPool,
                blockReward,
                3, // Each data row is a 3-hour period
            )
            
            totalProfit += periodProfit
        }
    }
    
    // 9. Calculate ROI (Return on Investment)
    // Assuming a rough cost of $50 per TH/s for mining hardware
    minerHashF := new(big.Float).SetInt(&miner.IndividualHashrate)
    hashTHFloat := new(big.Float).Quo(minerHashF, big.NewFloat(1e12))
    hashTH, _ := hashTHFloat.Float64()
    investmentCost := hashTH * 50 // $50 per TH/s
    
    var roi float64 = 0
    if investmentCost > 0 {
        roi = (totalProfit / investmentCost) * 100 // ROI as percentage
    }
    
    // Get the actual timeframe used (start and end dates) for reporting
    startDate := dataRows[startIdx].Timestamp
    endDate := dataRows[startIdx+periodsNeeded-1].Timestamp
    
    // 10. Return simulation result
    return t.SimulationResult{
        MinerType:    miner.MinerType.Name,
        Hashrate:     hashTH,
        Strategy:     strategy,
        TimeInterval: timeInterval,
        Region:       region.Name,
        NetProfit:    totalProfit,
        ROI:          roi,
        StartDate:    startDate,
        EndDate:      endDate,
    }
}

// Function to run multiple simulations and collect statistics
// Function to run multiple simulations and collect statistics
func runMonteCarloSimulations(dataRows []t.DataRow, numSimulations int, r *rand.Rand) []t.SimulationResult {
    results := make([]t.SimulationResult, numSimulations)
    
    // Run simulations
    for i := range make([]struct{}, numSimulations) {
        results[i] = runSimulation(dataRows, r)
        
        // Print progress every 1000 simulations
        if i%1000 == 0 && i > 0 {
            fmt.Printf("Completed %d of %d simulations\n", i, numSimulations)
        }
    }
    
    return results
}


// Function to analyze simulation results
func analyzeResults(results []t.SimulationResult) {
    // Group results by miner type
    minerTypeResults := make(map[string][]t.SimulationResult)
    for _, result := range results {
        minerTypeResults[result.MinerType] = append(minerTypeResults[result.MinerType], result)
    }
    
    // Group results by strategy
    strategyResults := make(map[t.Strategy][]t.SimulationResult)
    for _, result := range results {
        strategyResults[result.Strategy] = append(strategyResults[result.Strategy], result)
    }
    
    // Group results by region
    regionResults := make(map[string][]t.SimulationResult)
    for _, result := range results {
        regionResults[result.Region] = append(regionResults[result.Region], result)
    }
    
    // Group results by time interval
    intervalResults := make(map[float64][]t.SimulationResult)
    for _, result := range results {
        intervalResults[result.TimeInterval] = append(intervalResults[result.TimeInterval], result)
    }
    
    // Print average profit and ROI by miner type
    fmt.Println("\nResults by Miner Type:")
    fmt.Println("=====================")
    for minerType, typeResults := range minerTypeResults {
        var totalProfit, totalROI float64
        profitableSims := 0
        
        for _, result := range typeResults {
            totalProfit += result.NetProfit
            totalROI += result.ROI
            if result.NetProfit > 0 {
                profitableSims++
            }
        }
        
        avgProfit := totalProfit / float64(len(typeResults))
        avgROI := totalROI / float64(len(typeResults))
        profitablePercentage := float64(profitableSims) / float64(len(typeResults)) * 100
        
        fmt.Printf("%s miners: Avg Profit: $%.2f, Avg ROI: %.2f%%, Profitable: %.1f%%\n", 
            minerType, avgProfit, avgROI, profitablePercentage)
    }
    
    // Print average profit and ROI by strategy
    fmt.Println("\nResults by Strategy:")
    fmt.Println("==================")
    strategyNames := []string{"Solo Mining", "Pool Mining", "No Mining", "Switching Pool"}
    for strategy, stratResults := range strategyResults {
        var totalProfit, totalROI float64
        profitableSims := 0
        
        for _, result := range stratResults {
            totalProfit += result.NetProfit
            totalROI += result.ROI
            if result.NetProfit > 0 {
                profitableSims++
            }
        }
        
        avgProfit := totalProfit / float64(len(stratResults))
        avgROI := totalROI / float64(len(stratResults))
        profitablePercentage := float64(profitableSims) / float64(len(stratResults)) * 100
        
        fmt.Printf("%s: Avg Profit: $%.2f, Avg ROI: %.2f%%, Profitable: %.1f%%\n", 
            strategyNames[int(strategy)], avgProfit, avgROI, profitablePercentage)
    }
    
    // Print average profit and ROI by region
    fmt.Println("\nResults by Region:")
    fmt.Println("================")
    for region, regResults := range regionResults {
        var totalProfit, totalROI float64
        profitableSims := 0
        
        for _, result := range regResults {
            totalProfit += result.NetProfit
            totalROI += result.ROI
            if result.NetProfit > 0 {
                profitableSims++
            }
        }
        
        avgProfit := totalProfit / float64(len(regResults))
        avgROI := totalROI / float64(len(regResults))
        profitablePercentage := float64(profitableSims) / float64(len(regResults)) * 100
        
        fmt.Printf("%s: Avg Profit: $%.2f, Avg ROI: %.2f%%, Profitable: %.1f%%\n", 
            region, avgProfit, avgROI, profitablePercentage)
    }
    
    // Print average profit and ROI by time interval
    fmt.Println("\nResults by Time Interval:")
    fmt.Println("=======================")
    intervalNames := map[float64]string{
        DailyInterval:   "Daily",
        WeeklyInterval:  "Weekly",
        MonthlyInterval: "Monthly",
        YearlyInterval:  "Yearly",
    }
    
    for interval, intResults := range intervalResults {
        var totalProfit, totalROI float64
        profitableSims := 0
        
        for _, result := range intResults {
            totalProfit += result.NetProfit
            totalROI += result.ROI
            if result.NetProfit > 0 {
                profitableSims++
            }
        }
        
        avgProfit := totalProfit / float64(len(intResults))
        avgROI := totalROI / float64(len(intResults))
        profitablePercentage := float64(profitableSims) / float64(len(intResults)) * 100
        
        intervalName := intervalNames[interval]
        fmt.Printf("%s: Avg Profit: $%.2f, Avg ROI: %.2f%%, Profitable: %.1f%%\n", 
            intervalName, avgProfit, avgROI, profitablePercentage)
    }
    
    // Find most profitable combinations
    type combinationKey struct {
        MinerType string
        Strategy  t.Strategy
        Region    string
    }
    
    combinationResults := make(map[combinationKey][]t.SimulationResult)
    for _, result := range results {
        key := combinationKey{
            MinerType: result.MinerType,
            Strategy:  result.Strategy,
            Region:    result.Region,
        }
        combinationResults[key] = append(combinationResults[key], result)
    }
    
    type combinationSummary struct {
        MinerType    string
        Strategy     string
        Region       string
        AvgProfit    float64
        AvgROI       float64
        ProfitablePC float64
        Count        int
    }
    
    var summaries []combinationSummary
    for key, comboResults := range combinationResults {
        // Only consider combinations with enough samples
        if len(comboResults) < 10 {
            continue
        }
        
        var totalProfit, totalROI float64
        profitableSims := 0
        
        for _, result := range comboResults {
            totalProfit += result.NetProfit
            totalROI += result.ROI
            if result.NetProfit > 0 {
                profitableSims++
            }
        }
        
        avgProfit := totalProfit / float64(len(comboResults))
        avgROI := totalROI / float64(len(comboResults))
        profitablePercentage := float64(profitableSims) / float64(len(comboResults)) * 100
        
        strategyName := strategyNames[int(key.Strategy)]
        
        summaries = append(summaries, combinationSummary{
            MinerType:    key.MinerType,
            Strategy:     strategyName,
            Region:       key.Region,
            AvgProfit:    avgProfit,
            AvgROI:       avgROI,
            ProfitablePC: profitablePercentage,
            Count:        len(comboResults),
        })
    }
    
    // Sort by average profit
    sort.Slice(summaries, func(i, j int) bool {
        return summaries[i].AvgProfit > summaries[j].AvgProfit
    })
    
    // Print top 10 most profitable combinations
    fmt.Println("\nTop 10 Most Profitable Combinations:")
    fmt.Println("==================================")
    fmt.Printf("%-12s %-15s %-12s %-12s %-10s %-12s\n", 
        "Miner Type", "Strategy", "Region", "Avg Profit", "Avg ROI", "Profitable%")
    
    end := min(10, len(summaries)) 
    if len(summaries) < 10 {
        end = len(summaries)
    }
    
    for _, summary := range summaries[:end] {
        fmt.Printf("%-12s %-15s %-12s $%-11.2f %-10.2f%% %-12.1f%%\n",
            summary.MinerType,
            summary.Strategy,
            summary.Region,
            summary.AvgProfit,
            summary.AvgROI,
            summary.ProfitablePC)
    }
    
    // Print bottom 5 least profitable combinations
    fmt.Println("\nBottom 5 Least Profitable Combinations:")
    fmt.Println("====================================")
    fmt.Printf("%-12s %-15s %-12s %-12s %-10s %-12s\n", 
        "Miner Type", "Strategy", "Region", "Avg Profit", "Avg ROI", "Profitable%")
    
    start := max(0, len(summaries)-5)
    
    for i := start; i < len(summaries); i++ {
        fmt.Printf("%-12s %-15s %-12s $%-11.2f %-10.2f%% %-12.1f%%\n",
            summaries[i].MinerType,
            summaries[i].Strategy,
            summaries[i].Region,
            summaries[i].AvgProfit,
            summaries[i].AvgROI,
            summaries[i].ProfitablePC)
    }
}

func main() {
    // Initialize a random number generator with a specific seed
    // instead of using the global one with deprecated Seed function
    r := rand.New(rand.NewSource(time.Now().UnixNano()))
    
    // Load data from Excel file
    dataRows, err := dataloader.LoadDataRows("bitcoin_metrics_and_energy_updated_final_manually_scaled.xlsx")
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Loaded %d data rows from Excel file\n", len(dataRows))
    
    // Number of simulations to run
    numSimulations := 10000
    
    fmt.Printf("Running %d Monte Carlo simulations...\n", numSimulations)
    
    // Run simulations - passing the random generator
    results := runMonteCarloSimulations(dataRows, numSimulations, r)
    
    // Analyze and display results
    analyzeResults(results)
    
    fmt.Println("\nSimulation complete.")
}
