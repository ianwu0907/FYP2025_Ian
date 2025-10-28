import pandas as pd
import numpy as np

def transform(df):
    """Transform messy spreadsheet data into normalized table."""
    result = df.copy()
    
    # STEP 0: Remove summary rows if detected
    summary_indices = [0, 1, 2, 3, 4, 5, 6, 7, 55, 56, 57, 58, 59, 60, 61, 62, 110, 111, 112, 113, 114, 115, 116, 117, 165, 166, 167, 168, 169, 170, 171, 172, 220, 221, 222, 223, 224, 225, 226, 227, 275, 276, 277, 278, 279, 280, 281, 282, 330, 331, 332, 333, 334, 335, 336, 337, 385, 386, 387, 388, 389, 390, 391, 392, 440, 441, 442, 443, 444, 445, 446, 447, 495, 496, 497, 498, 499, 500, 501, 502, 550, 551, 552, 553, 554, 555, 556, 557, 605, 606, 607, 608, 609, 610, 611, 612, 660, 661, 662, 663, 664, 665, 666, 667, 715, 716, 717, 718, 719, 720, 721, 722, 770, 771, 772, 773, 774, 775, 776, 777, 825, 826, 827, 828, 829, 830, 831, 832, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1225]
    if summary_indices:
        try:
            result = result.drop(index=summary_indices).reset_index(drop=True)
            print(f"Removed {len(summary_indices)} summary rows")
        except Exception as e:
            print(f"Warning: Could not remove summary rows: {e}")

    # STEP 1: Split composite columns
    # Split 項目
    try:
        split_result = result['項目'].str.split(' - ', expand=True)
        if split_result.shape[1] >= 2:
            result['item_type'] = split_result[0]
            result['gender'] = split_result[1]
            print(f'Split 項目 into 2 columns')
        else:
            print(f'Warning: Split of 項目 produced {split_result.shape[1]} columns, expected 2')
    except Exception as e:
        print(f'Error splitting 項目: {e}')

    # Split Item
    try:
        split_result = result['Item'].str.split(' - ', expand=True)
        if split_result.shape[1] >= 2:
            result['item_type_en'] = split_result[0]
            result['gender_en'] = split_result[1]
            print(f'Split Item into 2 columns')
        else:
            print(f'Warning: Split of Item produced {split_result.shape[1]} columns, expected 2')
    except Exception as e:
        print(f'Error splitting Item: {e}')

    # STEP 2: Type conversions
    # Define conversion helper functions
    def convert_reported_to_police(value):
        '''Convert 有否向警方舉報事件 to boolean with N/A handling
        Based on metadata unique_values: ['不適用', '有', '沒有']
        '''
        if pd.isna(value):
            return None
        value_str = str(value).strip()
        
        # True values
        if value_str in ['有']:
            return True
        # False values
        elif value_str in ['沒有']:
            return False
        # N/A values
        elif value_str in ['不適用']:
            return None
        else:
            print(f"Warning: Unexpected value in 有否向警方舉報事件: '{value}'")
            return None

    def convert_incident_reported(value):
        '''Convert Incident Being or Not Reported to Police to boolean
        Based on metadata unique_values: ['Yes', 'No']
        '''
        if pd.isna(value):
            return None
        value_str = str(value).strip()
        
        # True values
        if value_str in ['Yes']:
            return True
        # False values
        elif value_str in ['No']:
            return False
        else:
            print(f"Warning: Unexpected value in Incident Being or Not Reported to Police: '{value}'")
            return None

    # Apply conversions
    try:
        result['有否向警方舉報事件'] = result['有否向警方舉報事件'].apply(convert_reported_to_police)
        print(f'Converted 有否向警方舉報事件 to boolean')
    except Exception as e:
        print(f'Error converting 有否向警方舉報事件: {e}')
    result = result.dropna(subset=[result.columns[0]]).reset_index(drop=True)
    
    # Step 1: Keep operations - copy and rename
    result['year'] = result['年份/Year']
    result['category_zh'] = result['類別']
    result['category_en'] = result['Category']
    result['cases'] = result['個案數字/No. of Cases']

    # Step 2: Split operations
    def split_func_1(text):
        if pd.isna(text) or text == '':
            return pd.Series(['' for _ in range(2)])
        text_str = str(text).strip()
        if ' - ' in text_str:
            parts = text_str.split(' - ', 1)
            if len(parts) >= 2:
                return pd.Series([p.strip() for p in parts[:2]])
            return pd.Series([parts[0].strip(), ''])
        return pd.Series([text_str, ''])
    
    result[['item_type_zh', 'item_gender_zh']] = result['項目'].apply(split_func_1)
    
    def split_func_2(text):
        if pd.isna(text) or text == '':
            return pd.Series(['' for _ in range(2)])
        text_str = str(text).strip()
        if ' - ' in text_str:
            parts = text_str.split(' - ', 1)
            if len(parts) >= 2:
                return pd.Series([p.strip() for p in parts[:2]])
            return pd.Series([parts[0].strip(), ''])
        return pd.Series([text_str, ''])
    
    result[['item_type_en', 'item_gender_en']] = result['Item'].apply(split_func_2)
    
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