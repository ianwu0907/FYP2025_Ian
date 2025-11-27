import pandas as pd
import numpy as np

def transform(df):
    """Transform messy spreadsheet to normalized table"""
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
    # Split 項目  # Metadata: 60% frequency, 2 parts
    try:
        split_result = result['項目'].str.split(' - ', expand=True)
        if split_result.shape[1] >= 2:
            result['item_type_zh'] = split_result[0]
            result['item_dimension_zh'] = split_result[1]
            print(f'Split 項目 into 2 columns')
        else:
            print(f'Warning: Split of 項目 produced {split_result.shape[1]} columns, expected 2')
    except Exception as e:
        print(f'Error splitting 項目: {e}')
    
    # Split Item  # Metadata: 60% frequency, 2 parts
    try:
        split_result = result['Item'].str.split(' - ', expand=True)
        if split_result.shape[1] >= 2:
            result['item_type_en'] = split_result[0]
            result['item_dimension_en'] = split_result[1]
            print(f'Split Item into 2 columns')
        else:
            print(f'Warning: Split of Item produced {split_result.shape[1]} columns, expected 2')
    except Exception as e:
        print(f'Error splitting Item: {e}')
    
    # STEP 2: Type conversions
    def convert_有否向警方舉報事件(value):
        '''
        Convert 有否向警方舉報事件 to boolean
        Based on metadata unique_values: ['不適用', '有', '沒有']
        Value mapping: {'不適用': None, '有': True, '沒有': False}
        '''
        if pd.isna(value):
            return None
        value_str = str(value).strip().lower()
        
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
    
    # Apply conversions
    try:
        result['有否向警方舉報事件'] = result['有否向警方舉報事件'].apply(convert_有否向警方舉報事件)
        print(f'Converted 有否向警方舉報事件 to boolean')
    except Exception as e:
        print(f'Error converting 有否向警方舉報事件: {e}')
    
    # STEP 3: Rename columns
    rename_map = {
        '年份/Year': 'year',
        '有否向警方舉報事件': 'incident_reported_to_police',
        '類別': 'category_zh',
        'Category': 'category_en',
        '個案數字/No. of Cases': 'number_of_cases'
    }
    result.rename(columns=rename_map, inplace=True)
    
    # STEP 4: Select and order final columns
    expected_columns = ['year', 'incident_reported_to_police', 'category_zh', 'category_en', 'item_type_zh', 'item_dimension_zh', 'item_type_en', 'item_dimension_en', 'number_of_cases']
    result = result[expected_columns]
    
    return result