import pandas as pd
import numpy as np

def transform(df):
    print(f"Input shape: {df.shape}")

    # Step 1: Slice to data region
    result = df.iloc[6:47].copy()
    result = result.reset_index(drop=True)
    print(f"After slicing to data region: {result.shape}")

    # Step 2: Remove blank rows
    result = result.dropna(how="all")
    print(f"After removing blank rows: {result.shape}")

    # Step 3: Merge bilingual alternating rows
    merged_rows = []
    data_rows = result.values.tolist()
    for i in range(0, len(data_rows), 2):
        cn_row = data_rows[i]
        en_row = data_rows[i+1] if i+1 < len(data_rows) else []
        
        if pd.notna(cn_row[1]) and str(cn_row[1]).strip():
            merged = {
                "ethnic_group_cn": cn_row[1],
                "ethnic_group_en": en_row[1] if en_row else ""
            }
            merged_rows.append(merged)        
    result_cn = pd.DataFrame(merged_rows)
    print(f"After merging bilingual rows: {result_cn.shape}")

    # Step 4: Prepare to unpivot year and marital status information
    marital_statuses = ["從未結婚", "已婚", "喪偶／離婚／分居"]
    years = ["2011"]
    sex_categories = ["男", "女"]
    
    records = []
    
    for i in range(len(result_cn)):
        base = result_cn.iloc[i]
        for j in range(len(marital_statuses)):
            for k, sex in enumerate(sex_categories):
                percentage = pd.to_numeric(result.iloc[i, 3 + j * 3 + k], errors='coerce')
                if not np.isnan(percentage):
                    records.append({
                        "ethnic_group_cn": base['ethnic_group_cn'],
                        "ethnic_group_en": base['ethnic_group_en'],
                        "marital_status": marital_statuses[j],
                        "year": years[0],
                        "sex": sex,
                        "percentage": percentage
                    })
    
    final_result = pd.DataFrame(records)
    print(f"After unpivoting: {final_result.shape}")

    # Step 5: Remove aggregation rows
    final_result = final_result[~final_result['ethnic_group_cn'].str.contains("所有少數族裔人士", na=False)]
    print(f"After removing aggregation rows: {final_result.shape}")

    # Final column order
    final_result = final_result[["ethnic_group_cn", "ethnic_group_en", "marital_status", "year", "sex", "percentage"]]
    print(f"Final output shape: {final_result.shape}")
    return final_result