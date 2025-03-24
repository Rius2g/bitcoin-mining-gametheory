package main

import (
	"fmt"
	"log"
	"math/big"
	"math/rand"
    "runtime"
    "strings"

	dataloader "models/dataloader"
	payout "models/payout"
	t "models/types"
	"sort"
	"sync"
	"time"
)

// Time intervals in hours
const (
    DailyInterval   = 24
    WeeklyInterval  = 24 * 7
    MonthlyInterval = 24 * 30
    YearlyInterval  = 24 * 365
)

// Current block reward (BTC)

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

// Function to select random time interval
func getRandomTimeInterval(r *rand.Rand) float64 {
    intervals := []float64{DailyInterval, WeeklyInterval, MonthlyInterval, YearlyInterval}
    return intervals[r.Intn(len(intervals))]
}

// Function to select random strategy
func getRandomStrategy(r *rand.Rand) t.Strategy {
    return t.Strategy(r.Intn(3))
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

// New function to create a miner of a specific type
func createMinerOfType(minerType t.MinerType, r *rand.Rand) t.Miner {
    // Convert hashrate to TH/s and then to H/s for big.Int representation
    hashrateHs := new(big.Int).SetInt64(int64(minerType.MinHash * 1e12))
    
    // Set power efficiency based on miner type
    // You could also make this fixed per miner type
    efficiencyWpTh := 30.0 // Fixed at 30 W/TH for consistency
    
    return t.Miner{
        PowerEfficiency:    efficiencyWpTh,
        IndividualHashrate: *hashrateHs,
        MinerType:          minerType,
    }
}


// Modified simulation function to take a specific miner type
func runSimulationWithMinerType(dataRows []t.DataRow, minerType t.MinerType, r *rand.Rand) t.SimulationResult {
    // 1. Create miner of the specified type
    miner := createMinerOfType(minerType, r)
    
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
        for i := 0; i < periodsNeeded; i++ {
            dataRow := dataRows[startIdx+i]
            electricityCost := getElectricityCost(region, dataRow)
            totalHashrate := float64ToBigInt(dataRow.ConvertedHash)

            
            // Calculate profit for this 3-hour period
            periodProfit := payout.CalculatePayout(
                dataRow.BTCPriceUSD,
                totalHashrate,
                miner,
                electricityCost,
                dataRow.TransactionFees,
                false, // Not in a pool
                t.MiningPool{}, // Empty pool (not used)
                dataRow.BlockReward,
            )
            
            totalProfit += periodProfit
        }
    
    case t.PoolMining:
        // Select a random pool for the whole period
        pool := getRandomPool(r)
        
        for i := 0; i < periodsNeeded; i++ {
            dataRow := dataRows[startIdx+i]
            electricityCost := getElectricityCost(region, dataRow)
            totalHashrate := float64ToBigInt(dataRow.ConvertedHash)
 
            
            // Calculate profit for this 3-hour period
            periodProfit := payout.CalculatePayout(
                dataRow.BTCPriceUSD,
                totalHashrate,
                miner,
                electricityCost,
                dataRow.TransactionFees,
                true, // In a pool
                pool,
                dataRow.BlockReward,
            )
            
            totalProfit += periodProfit
        }
    
    
    case t.SwitchingPool:
        // Switch pools halfway through the time interval
        pool1 := getRandomPool(r)
        pool2 := getRandomPool(r)
        
        switchPoint := periodsNeeded / 2
        
        for i := 0; i < periodsNeeded; i++ {
            dataRow := dataRows[startIdx+i]
            electricityCost := getElectricityCost(region, dataRow)

            totalHashrate := float64ToBigInt(dataRow.ConvertedHash)

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
                dataRow.BlockReward,
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

// Function to run batch of simulations for a specific miner type
func runBatchSimulations(dataRows []t.DataRow, numSimulations int, minerType t.MinerType, r *rand.Rand) []t.SimulationResult {
    results := make([]t.SimulationResult, numSimulations)
    
    // Use parallel processing with goroutines
    maxConcurrency := runtime.NumCPU()
    semaphore := make(chan struct{}, maxConcurrency)

    var progressMutex sync.Mutex 
    completedSimulations := 0 

    var wg sync.WaitGroup 

    for i := range results {
        wg.Add(1)

        go func(index int) {
            defer wg.Done()

            semaphore <- struct{}{} // Acquire semaphore 
            defer func(){ <-semaphore }()

            localRand := rand.New(rand.NewSource(r.Int63())) // Create local random generator

            results[index] = runSimulationWithMinerType(dataRows, minerType, localRand) 

            progressMutex.Lock() 
            completedSimulations++ 
            if completedSimulations%1000 == 0 {
                fmt.Printf("Completed %d of %d simulations for %s miners\n", 
                           completedSimulations, numSimulations, minerType.Name)
            }
            progressMutex.Unlock()
        }(i)
    }
    
    wg.Wait()
    return results
}

// Function to run simulations for all miner types
func runAllMinerTypeSimulations(dataRows []t.DataRow, simulationsPerType int, r *rand.Rand) map[string][]t.SimulationResult {
    // Map to store results by miner type
    allResults := make(map[string][]t.SimulationResult)
    
    // For each miner type, run a batch of simulations
    for _, minerType := range t.MinerTypes {
        fmt.Printf("\nRunning %d simulations for %s miners (%.1f-%.1f TH/s)...\n", 
                   simulationsPerType, minerType.Name, minerType.MinHash, minerType.MaxHash)
        
        startTime := time.Now()
        
        // Run simulations for this miner type
        results := runBatchSimulations(dataRows, simulationsPerType, minerType, r)
        
        elapsed := time.Since(startTime)
        fmt.Printf("Completed simulations for %s miners in %s\n", minerType.Name, elapsed)
        
        // Store results
        allResults[minerType.Name] = results
        
        // Optional: Garbage collect between batches to free memory
        runtime.GC()
    }
    
    return allResults
}

// Helper for calculating normalized metrics
type normalizedMetrics struct {
    AvgProfit          float64
    AvgProfitPerTH     float64
    AvgROI             float64
    ProfitablePercent  float64
    Count              int
}

// Calculate normalized metrics for a set of results
func calculateNormalizedMetrics(results []t.SimulationResult) normalizedMetrics {
    var totalProfit, totalProfitPerTH, totalROI float64
    profitable := 0
    count := len(results)
    
    if count == 0 {
        return normalizedMetrics{}
    }
    
    for _, result := range results {
        totalProfit += result.NetProfit
        
        // Calculate profit per TH/s
        if result.Hashrate > 0 {
            profitPerTH := result.NetProfit / result.Hashrate
            totalProfitPerTH += profitPerTH
        }
        
        totalROI += result.ROI
        
        if result.NetProfit > 0 {
            profitable++
        }
    }
    
    return normalizedMetrics{
        AvgProfit:         totalProfit / float64(count),
        AvgProfitPerTH:    totalProfitPerTH / float64(count),
        AvgROI:            totalROI / float64(count),
        ProfitablePercent: (float64(profitable) / float64(count)) * 100,
        Count:             count,
    }
}

// Function to analyze results by miner type with normalized metrics
func analyzeResultsByMinerType(allResults map[string][]t.SimulationResult) {
    // Strategy names for display
    strategyNames := []string{"Solo Mining", "Pool Mining", "Switching Pool"}
    
    // Process each miner type
    for minerType, results := range allResults {
        fmt.Printf("\n\n========== RESULTS FOR %s MINERS ==========\n", minerType)
        
        // Get hashrate range for this miner type
        var hashrate float64 = -1 
        if (len(results) > 0) {
            hashrate = results[0].Hashrate 
        }
        var minHashrate, maxHashrate float64 = -1, -1
        for _, result := range results {
            if minHashrate == -1 || result.Hashrate < minHashrate {
                minHashrate = result.Hashrate
            }
            if maxHashrate == -1 || result.Hashrate > maxHashrate {
                maxHashrate = result.Hashrate
            }
        }
        fmt.Printf("Hashrate: %.1f TH/s\n", hashrate)    
        fmt.Printf("Total Simulations: %d\n\n", len(results))
        
        // Overall metrics for this miner type
        metrics := calculateNormalizedMetrics(results)
        fmt.Printf("Overall Performance:\n")
        fmt.Printf("  Avg Profit: $%.2f\n", metrics.AvgProfit)
        fmt.Printf("  Avg Profit per TH/s: $%.2f\n", metrics.AvgProfitPerTH)
        fmt.Printf("  Avg ROI: %.2f%%\n", metrics.AvgROI)
        fmt.Printf("  Profitable Simulations: %.1f%%\n\n", metrics.ProfitablePercent)
        
        // Group by strategy
        strategyResults := make(map[t.Strategy][]t.SimulationResult)
        for _, result := range results {
            strategyResults[result.Strategy] = append(strategyResults[result.Strategy], result)
        }
        
        // Print strategy performance
        fmt.Printf("Performance by Strategy:\n")
        fmt.Printf("%-15s %-12s %-15s %-10s %-15s\n", 
                 "Strategy", "Avg Profit", "Profit/TH/s", "ROI %", "Profitable %")
        fmt.Println(strings.Repeat("-", 70))
        
        for strategy, stratResults := range strategyResults {
            metrics := calculateNormalizedMetrics(stratResults)
            fmt.Printf("%-15s $%-11.2f $%-14.2f %-10.2f%% %-15.1f%%\n", 
                     strategyNames[int(strategy)], 
                     metrics.AvgProfit, 
                     metrics.AvgProfitPerTH,
                     metrics.AvgROI, 
                     metrics.ProfitablePercent)
        }
        
        // Group by region
        regionResults := make(map[string][]t.SimulationResult)
        for _, result := range results {
            regionResults[result.Region] = append(regionResults[result.Region], result)
        }
        
        // Print region performance
        fmt.Printf("\nPerformance by Region:\n")
        fmt.Printf("%-12s %-12s %-15s %-10s %-15s\n", 
                 "Region", "Avg Profit", "Profit/TH/s", "ROI %", "Profitable %")
        fmt.Println(strings.Repeat("-", 70))
        
        for region, regResults := range regionResults {
            metrics := calculateNormalizedMetrics(regResults)
            fmt.Printf("%-12s $%-11.2f $%-14.2f %-10.2f%% %-15.1f%%\n", 
                     region, 
                     metrics.AvgProfit, 
                     metrics.AvgProfitPerTH,
                     metrics.AvgROI, 
                     metrics.ProfitablePercent)
        }
        
        // Find best combinations for this miner type
        fmt.Printf("\nTop 5 Most Profitable Combinations for %s Miners:\n", minerType)
        findTopCombinations(results, 5, strategyNames)
    }
}

// Helper function to find top combinations
func findTopCombinations(results []t.SimulationResult, topN int, strategyNames []string) {
    type combinationKey struct {
        Strategy t.Strategy
        Region   string
    }
    
    combinationResults := make(map[combinationKey][]t.SimulationResult)
    
    for _, result := range results {
        key := combinationKey{
            Strategy: result.Strategy,
            Region:   result.Region,
        }
        combinationResults[key] = append(combinationResults[key], result)
    }
    
    type combinationSummary struct {
        Strategy         string
        Region           string
        AvgProfit        float64
        AvgProfitPerTH   float64
        AvgROI           float64
        ProfitablePC     float64
        Count            int
    }
    
    var summaries []combinationSummary
    
    for key, comboResults := range combinationResults {
        // Only consider combinations with enough samples
        if len(comboResults) < 5 {
            continue
        }
        
        metrics := calculateNormalizedMetrics(comboResults)
        
        summaries = append(summaries, combinationSummary{
            Strategy:       strategyNames[int(key.Strategy)],
            Region:         key.Region,
            AvgProfit:      metrics.AvgProfit,
            AvgProfitPerTH: metrics.AvgProfitPerTH,
            AvgROI:         metrics.AvgROI,
            ProfitablePC:   metrics.ProfitablePercent,
            Count:          metrics.Count,
        })
    }
    
    // Sort by profit per TH/s (normalized metric for fair comparison)
    sort.Slice(summaries, func(i, j int) bool {
        return summaries[i].AvgProfitPerTH > summaries[j].AvgProfitPerTH
    })
    
    // Print header
    fmt.Printf("%-15s %-12s %-15s %-15s %-10s %-12s\n", 
             "Strategy", "Region", "Avg Profit", "Profit/TH/s", "ROI %", "Profitable %")
    fmt.Println(strings.Repeat("-", 80))
    
    // Print top N
    end := min(topN, len(summaries))
    for i := 0; i < end; i++ {
        summary := summaries[i]
        fmt.Printf("%-15s %-12s $%-13.2f $%-13.2f %-10.2f%% %-12.1f%%\n",
                 summary.Strategy,
                 summary.Region,
                 summary.AvgProfit,
                 summary.AvgProfitPerTH,
                 summary.AvgROI,
                 summary.ProfitablePC)
    }
}

// Modified main function to use the new batched simulation approach
func main() {
    // Initialize a random number generator with a specific seed
    r := rand.New(rand.NewSource(time.Now().UnixNano()))

    // Set GOMAXPROCS to use all available CPU cores
    runtime.GOMAXPROCS(runtime.NumCPU())
    fmt.Printf("Running on %d CPU cores\n", runtime.NumCPU())
    
    // Load data from Excel file
    dataRows, err := dataloader.LoadDataRows("bitcoin_metrics_and_energy_updated_final_manually_scaled.xlsx")
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Loaded %d data rows from Excel file\n", len(dataRows))
    
    simulationsPerType := 300000 
    
    fmt.Printf("Running %d simulations for each miner type...\n", simulationsPerType)
    
    startTime := time.Now()
    
    // Run simulations for all miner types
    allResults := runAllMinerTypeSimulations(dataRows, simulationsPerType, r)
    
    elapsedTime := time.Since(startTime)
    fmt.Printf("\nAll simulations completed in %s\n", elapsedTime)
    
    // Analyze and display results by miner type
    analyzeResultsByMinerType(allResults)
    
    fmt.Println("\nSimulation complete.")
}

func float64ToBigInt(val float64) *big.Int {
    str := fmt.Sprintf("%.0f", val)
    b := new(big.Int)
    b.SetString(str, 10)
    return b
}

