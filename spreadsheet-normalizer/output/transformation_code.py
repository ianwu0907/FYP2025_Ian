import pandas as pd
import numpy as np

def transform(df):
    result = df.copy()
    
    # STEP 0: Handle implicit aggregation (if detected)
    summary_indices = [0, 1, 2, 3, 4, 5, 6, 7, 55, 56, 57, 58, 59, 60, 61, 62, 
                       110, 111, 112, 113, 114, 115, 116, 117, 165, 166, 167, 
                       168, 169, 170, 171, 172, 220, 221, 222, 223, 224, 225, 
                       226, 227, 275, 276, 277, 278, 279, 280, 281, 282, 330, 
                       331, 332, 333, 334, 335, 336, 337, 385, 386, 387, 388, 
                       389, 390, 391, 392, 440, 441, 442, 443, 444, 445, 446, 
                       447, 495, 496, 497, 498, 499, 500, 501, 502, 550, 551, 
                       552, 553, 554, 555, 556, 557, 605, 606, 607, 608, 609, 
                       610, 611, 612, 660, 661, 662, 663, 664, 665, 666, 667, 
                       715, 716, 717, 718, 719, 720, 721, 722, 770, 771, 772, 
                       773, 774, 775, 776, 777, 825, 826, 827, 828, 829, 830, 
                       831, 832, 880, 881, 882, 883, 884, 885, 886, 887, 888, 
                       889, 890, 891, 892, 893, 894, 895, 990, 991, 992, 993, 
                       994, 995, 996, 997, 998, 999, 1000, 1001, 1002, 1003, 
                       1004, 1005, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 
                       1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 
                       1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 
                       1219, 1220, 1221, 1222, 1223, 1224, 1225]
    
    try:
        result = result.drop(index=summary_indices).reset_index(drop=True)
    except Exception as e:
        print(f"Error removing summary rows: {e}")
    
    # STEP 1: Apply split operations
    try:
        result[['item_type_zh', 'gender_zh']] = result['項目'].str.split(' - ', expand=True)
        result[['item_type_en', 'gender_en']] = result['Item'].str.split(' - ', expand=True)
    except Exception as e:
        print(f"Error during split operations: {e}")
    
    # STEP 2: Handle bilingual columns
    try:
        result['year'] = result['年份/Year'].astype(int)
        result['reported_to_police'] = result['有否向警方舉報事件'].apply(lambda x: x != '不適用')
        result['category_zh'] = result['類別']
        result['category_en'] = result['Category']
        result['number_of_cases'] = result['個案數字/No. of Cases'].fillna(0).astype(int)
    except Exception as e:
        print(f"Error during column handling: {e}")
    
    # STEP 3: Column selection and ordering
    expected_output_columns = ['year', 'reported_to_police', 'category_zh', 'category_en', 
                               'item_type_zh', 'gender_zh', 'item_type_en', 'gender_en', 
                               'number_of_cases']
    
    try:
        result = result[expected_output_columns]
    except KeyError as e:
        print(f"Missing columns in result: {e}")
    
    return result

# Execute transformation
result_df = transform(df)