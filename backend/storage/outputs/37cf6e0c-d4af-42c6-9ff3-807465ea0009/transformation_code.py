import pandas as pd
import numpy as np

def transform(df):
    """Transform messy spreadsheet to normalized table"""
    result = df.copy()

    # STEP 0: Remove summary rows if detected
    summary_indices = [0, 1, 2, 3, 4, 5, 6, 7, 55, 56, 57, 58, 59, 60, 61, 62, 110, 111, 112, 113, 114, 115, 116, 117, 165, 166, 167, 168, 169, 170, 171, 172, 220, 221, 222, 223, 224, 225, 226, 227, 275, 276, 277, 278, 279, 280, 281, 282, 330, 331, 332, 333, 334, 335, 336, 337, 385, 386, 387, 388, 389, 390, 391, 392, 440, 441, 442, 443, 444, 445, 446, 447, 495, 496, 497, 498, 499, 500, 501, 502, 550, 551, 552, 553, 554, 555, 556, 557, 605, 606, 607, 608, 609, 610, 611, 612, 660, 661, 662, 663, 664, 665, 666, 667, 715, 716, 717, 718, 719, 720, 721, 722, 770, 771, 772, 773, 774, 775, 776, 777, 825, 826, 827, 828, 829, 830, 831, 832, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1225]
    
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
            result['abuse_type_zh'] = split_result[0]
            result['gender_zh'] = split_result[1]
            print(f'Split 項目 into 2 columns')
        else:
            result['abuse_type_zh'] = result['項目'] # If split fails, keep original value
            result['gender_zh'] = None
            print(f'Warning: Split of 項目 produced {split_result.shape[1]} columns, expected 2. Keeping original and setting gender_zh to None')
    except Exception as e:
        print(f'Error splitting 項目: {e}')
        result['abuse_type_zh'] = result['項目'] # If split fails, keep original value
        result['gender_zh'] = None

    # Split Item  # Metadata: 60% frequency, 2 parts
    try:
        split_result = result['Item'].str.split(' - ', expand=True)
        if split_result.shape[1] >= 2:
            result['abuse_type_en'] = split_result[0]
            result['gender_en'] = split_result[1]
            print(f'Split Item into 2 columns')
        else:
            result['abuse_type_en'] = result['Item'] # If split fails, keep original value
            result['gender_en'] = None
            print(f'Warning: Split of Item produced {split_result.shape[1]} columns, expected 2. Keeping original and setting gender_en to None')
    except Exception as e:
        print(f'Error splitting Item: {e}')
        result['abuse_type_en'] = result['Item'] # If split fails, keep original value
        result['gender_en'] = None

    # STEP 2: Type conversions

    # Define conversion helper functions
    def safe_convert_categorical(series, mapping):
        """Safely converts a series to categorical based on a mapping."""
        try:
            # Normalize strings before mapping
            series = series.astype(str).str.strip().str.lower().str.replace('　', ' ') # full-width space
            return series.map(mapping).astype('category')
        except Exception as e:
            print(f"Warning: Categorical conversion failed: {e}")
            return series  # Return original series if conversion fails

    # Apply conversions

    # reported_to_police_zh: categorical, mapping: {'不適用': 'N/A', '有': 'Yes', '沒有': 'No'}
    mapping_reported_to_police_zh = {'不適用': 'N/A', '有': 'Yes', '沒有': 'No'}
    result['reported_to_police_zh'] = safe_convert_categorical(result['有否向警方舉報事件'], mapping_reported_to_police_zh)

    # reported_to_police_en: categorical, mapping: {'Yes': 'Yes', 'No': 'No'}
    mapping_reported_to_police_en = {'yes': 'Yes', 'no': 'No'} # Add lowercase variants
    result['reported_to_police_en'] = safe_convert_categorical(result['Incident Being or Not Reported to Police'], mapping_reported_to_police_en)

    # category_zh: categorical, mapping: {'虐待長者性質': 'Type of Elder Abuse', '虐待長者性質及受虐長者性別': 'Type of Elder Abuse and Sex of Elderly Person Being Abused', '施虐者與受虐長者關係': "Abuser's Relationship with Elderly Person Being Abused", '受虐長者居住地區': 'Residential District of Elderly Person Being Abused'}
    mapping_category_zh = {'虐待長者性質': 'Type of Elder Abuse', '虐待長者性質及受虐長者性別': 'Type of Elder Abuse and Sex of Elderly Person Being Abused', '施虐者與受虐長者關係': "Abuser's Relationship with Elderly Person Being Abused", '受虐長者居住地區': 'Residential District of Elderly Person Being Abused'}
    result['category_zh'] = safe_convert_categorical(result['類別'], mapping_category_zh)

    # category_en: categorical, mapping: {'Type of Elder Abuse': 'Type of Elder Abuse', 'Type of Elder Abuse and Sex of Elderly Person Being Abused': 'Type of Elder Abuse and Sex of Elderly Person Being Abused', "Abuser's Relationship with Elderly Person Being Abused": "Abuser's Relationship with Elderly Person Being Abused", 'Residential District of Elderly Person Being Abused': 'Residential District of Elderly Person Being Abused'}
    mapping_category_en = {'type of elder abuse': 'Type of Elder Abuse', 'type of elder abuse and sex of elderly person being abused': 'Type of Elder Abuse and Sex of Elderly Person Being Abused', "abuser's relationship with elderly person being abused": "Abuser's Relationship with Elderly Person Being Abused", 'residential district of elderly person being abused': 'Residential District of Elderly Person Being Abused'} # Add lowercase variants
    result['category_en'] = safe_convert_categorical(result['Category'], mapping_category_en)

    # year: integer
    try:
        result['year'] = pd.to_numeric(result['年份/Year'], errors='coerce').astype('Int64')
        print("Converted 'year' to integer")
    except Exception as e:
        print(f"Warning: Could not convert 'year' to integer: {e}")
        result['year'] = None

    # number_of_cases: integer
    try:
        result['number_of_cases'] = pd.to_numeric(result['個案數字/No. of Cases'], errors='coerce').astype('Int64')
        print("Converted 'number_of_cases' to integer")
    except Exception as e:
        print(f"Warning: Could not convert 'number_of_cases' to integer: {e}")
        result['number_of_cases'] = None

    # STEP 3: Rename columns
    rename_map = {
        '年份/Year': 'year',
        '有否向警方舉報事件': 'reported_to_police_zh',
        'Incident Being or Not Reported to Police': 'reported_to_police_en',
        '類別': 'category_zh',
        'Category': 'category_en',
        '個案數字/No. of Cases': 'number_of_cases'
    }
    result.rename(columns=rename_map, inplace=True)

    # STEP 4: Select and order final columns
    expected_columns = ['year', 'reported_to_police_zh', 'reported_to_police_en', 'category_zh', 'category_en', 'abuse_type_zh', 'gender_zh', 'abuse_type_en', 'gender_en', 'number_of_cases']
    # Ensure all expected columns exist, filling with None if missing
    for col in expected_columns:
        if col not in result.columns:
            result[col] = None
            print(f"Warning: Column '{col}' not found, filled with None")
    result = result[expected_columns]

    return result