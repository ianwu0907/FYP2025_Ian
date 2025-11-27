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

    # STEP 1: Rename columns
    rename_map = {
        '年份/Year': 'year',
        '有否向警方舉報事件': 'reported_to_police',
        'Incident Being or Not Reported to Police': 'incident_reported',
        '類別': 'category_zh',
        'Category': 'category_en',
        '項目': 'item_zh',
        'Item': 'item_en',
        '個案數字/No. of Cases': 'number_of_cases'
    }
    result.rename(columns=rename_map, inplace=True)

    # STEP 2: Normalize 'year' column
    try:
        result['year'] = pd.to_numeric(result['year'], errors='coerce')
    except Exception as e:
        print(f"Warning: Could not convert 'year' to numeric: {e}")

    # STEP 3: Normalize 'reported_to_police' column
    try:
        result['reported_to_police'] = result['reported_to_police'].replace({
            '有': True,
            '沒有': False,
            '不適用': np.nan
        }).astype('boolean')
    except Exception as e:
        print(f"Warning: Could not normalize 'reported_to_police': {e}")

    # STEP 4: Normalize 'incident_reported' column
    try:
        result['incident_reported'] = result['incident_reported'].replace({
            'Yes': True,
            'No': False
        }).astype('boolean')
    except Exception as e:
        print(f"Warning: Could not normalize 'incident_reported': {e}")

    # STEP 5: Normalize categorical columns
    try:
        result['category_zh'] = result['category_zh'].astype('category')
        result['category_en'] = result['category_en'].astype('category')
    except Exception as e:
        print(f"Warning: Could not normalize categorical columns: {e}")

    # STEP 6: Normalize 'item_zh' and 'item_en' columns
    try:
        result['item_zh'] = result['item_zh'].astype(str).str.strip()
        result['item_en'] = result['item_en'].astype(str).str.strip()
    except Exception as e:
        print(f"Warning: Could not normalize 'item_zh' or 'item_en': {e}")

    # STEP 7: Normalize 'number_of_cases' column
    try:
        result['number_of_cases'] = pd.to_numeric(result['number_of_cases'], errors='coerce')
    except Exception as e:
        print(f"Warning: Could not convert 'number_of_cases' to numeric: {e}")

    # STEP 8: Select and order final columns
    expected_columns = ['year', 'reported_to_police', 'incident_reported', 'category_zh', 'category_en', 'item_zh', 'item_en', 'number_of_cases']
    result = result[expected_columns]
    
    return result