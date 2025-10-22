import pandas as pd
import numpy as np

def transform(df):
    # Make a copy
    result = df.copy()
    
    # Step 1: Remove metadata rows
    try:
        result = result[result['年份/Year'].notna() & (result['年份/Year'] != '')].reset_index(drop=True)
    except Exception as e:
        print(f"Error removing metadata rows: {e}")
    
    # Step 2: Apply split operations
    try:
        # Split the '項目' column for Chinese
        item_split_cn = result['項目'].str.split(' - ', expand=True)
        result['item_type_chinese'] = item_split_cn[0].str.strip()
        result['gender_chinese'] = item_split_cn[1].str.strip() if item_split_cn.shape[1] > 1 else ''
        
        # Split the 'Item' column for English
        item_split_en = result['Item'].str.split(' - ', expand=True)
        result['item_type_english'] = item_split_en[0].str.strip()
        result['gender_english'] = item_split_en[1].str.strip() if item_split_en.shape[1] > 1 else ''
    except Exception as e:
        print(f"Error during split operations: {e}")
    
    # Step 3: Handle bilingual columns
    try:
        result['category_chinese'] = result['類別'].str.strip()
        result['category_english'] = result['Category'].str.strip()
    except Exception as e:
        print(f"Error handling bilingual columns: {e}")
    
    # Step 4: Type conversions
    try:
        result['year'] = pd.to_numeric(result['年份/Year'], errors='coerce').fillna(0).astype(int)
        result['reported_to_police'] = result['有否向警方舉報事件'].map({'適用': True, '不適用': False}).fillna(False)
        result['number_of_cases'] = pd.to_numeric(result['個案數字/No. of Cases'], errors='coerce').fillna(0).astype(int)
    except Exception as e:
        print(f"Error during type conversions: {e}")
    
    # Step 5: Reorder columns to match expected output
    expected_cols = ['year', 'reported_to_police', 'category_chinese', 'category_english', 'item_type_chinese', 'item_type_english', 'gender_chinese', 'gender_english', 'number_of_cases']
    
    # Step 6: Validation
    for col in expected_cols:
        if col not in result.columns:
            result[col] = ''
    
    # Return only expected columns in correct order
    result = result[expected_cols]
    
    return result

# Execute
# Assuming df is already defined and contains the input data
result_df = transform(df)