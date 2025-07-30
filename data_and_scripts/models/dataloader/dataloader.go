package dataloader

import (
    "fmt"
    "strconv"
    "github.com/xuri/excelize/v2"
    t "models/types"
)

func LoadDataRows(filePath string) ([]t.DataRow, error) {
    f, err := excelize.OpenFile(filePath)
    if err != nil {
        return nil, fmt.Errorf("failed to open Excel file: %w", err)
    }
    defer func() {
        _ = f.Close()
    }()
    
    // Get all rows from the first sheet
    rows, err := f.GetRows("Sheet1")
    if err != nil {
        return nil, fmt.Errorf("failed to get rows from sheet: %w", err)
    }
    
    var dataRows []t.DataRow
    
    // Skip the header row (i==0)
    for i, row := range rows {
        if i == 0 || len(row) < 12 {
            continue // Skip header or incomplete rows
        }
        
        dr := t.DataRow{}
        
        // Based on the Excel columns we observed
        dr.Timestamp = row[0]
        dr.ConvertedHash = parseFloatOrZero(row[2])
        dr.BTCPriceUSD = parseFloatOrZero(row[3])
        dr.TransactionFees = parseFloatOrZero(row[4])
        dr.TransactionFeesUSD = parseFloatOrZero(row[5])
        dr.ExchangeRate = parseFloatOrZero(row[6])
        dr.DynamicExchangeRate = parseFloatOrZero(row[7])
        dr.DenmarkPriceKWh = parseFloatOrZero(row[8])
        dr.TexasPriceKWh = parseFloatOrZero(row[9])
        dr.KazakhstanPriceKWh = parseFloatOrZero(row[10])
        dr.ChinaPriceKWh = parseFloatOrZero(row[11])
        
        // For block reward, check if it exists in the row
        if len(row) > 12 {
            dr.BlockReward = parseFloatOrZero(row[12])
        } else {
            // Default block reward if not available
            dr.BlockReward = 6.25
        }
        
        dataRows = append(dataRows, dr)
    }
    
    if len(dataRows) == 0 {
        return nil, fmt.Errorf("no valid data rows found in Excel file")
    }
    
    return dataRows, nil
}

func parseFloatOrZero(s string) float64 {
    val, err := strconv.ParseFloat(s, 64)
    if err != nil {
        return 0.0
    }
    return val
}
