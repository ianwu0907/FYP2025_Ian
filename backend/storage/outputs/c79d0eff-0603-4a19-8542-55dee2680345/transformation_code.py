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
            result['item_type_zh'] = split_result[0]
            result['gender_zh'] = split_result[1]
            print(f'Split 項目 into 2 columns')
        else:
            print(f'Warning: Split of 項目 produced {split_result.shape[1]} columns, expected 2')
    except Exception as e:
        print(f'Error splitting 項目: {e}')

    # Split Item  # Metadata: 60% frequency, 2 parts
    try:
        split_result = result['Item'].str.split(' - ', expand=True)
        if split_result.shape[1] >= 2:
            result['item_type'] = split_result[0]
            result['gender'] = split_result[1]
            print(f'Split Item into 2 columns')
        else:
            print(f'Warning: Split of Item produced {split_result.shape[1]} columns, expected 2')
    except Exception as e:
        print(f'Error splitting Item: {e}')


    # STEP 2: Type conversions

    # Define conversion helper functions

    def convert_reported_to_police(value):
        '''
        Convert reported_to_police to boolean
        Based on metadata unique_values: ['Yes', 'No']
        Value mapping: {'Yes': True, 'No': False}
        '''
        if pd.isna(value):
            return None
        value_str = str(value).strip().lower()
        
        # True values
        if value_str in ['yes']:
            return True
        # False values
        elif value_str in ['no']:
            return False
        # N/A values
        elif value_str in []:
            return None
        else:
            print(f"Warning: Unexpected value in reported_to_police: '{value}'")
            return None

    def convert_reported_to_police_zh(value):
        '''
        Convert reported_to_police_zh to categorical
        Based on metadata unique_values: ['不適用', '有', '沒有']
        Value mapping: {'有': 'yes', '沒有': 'no', '不適用': 'not_applicable'}
        '''
        if pd.isna(value):
            return None
        value_str = str(value).strip().lower()
        value_str = value_str.replace('\u3000', ' ')  # Replace full-width spaces
        
        if value_str == '有':
            return 'yes'
        elif value_str == '沒有':
            return 'no'
        elif value_str == '不適用':
            return 'not_applicable'
        else:
            print(f"Warning: Unexpected value in reported_to_police_zh: '{value}'")
            return None

    def convert_category(value):
        '''
        Convert category to categorical
        Based on metadata unique_values: ['Type of Elder Abuse', 'Type of Elder Abuse and Sex of Elderly Person Being Abused', "Abuser's Relationship with Elderly Person Being Abused", 'Residential District of Elderly Person Being Abused']
        Value mapping: {'Type of Elder Abuse': 'type_of_elder_abuse', 'Type of Elder Abuse and Sex of Elderly Person Being Abused': 'type_of_elder_abuse_and_sex', "Abuser's Relationship with Elderly Person Being Abused": 'abusers_relationship', 'Residential District of Elderly Person Being Abused': 'residential_district'}
        '''
        if pd.isna(value):
            return None
        value_str = str(value).strip().lower()
        value_str = value_str.replace('\u3000', ' ')  # Replace full-width spaces

        if value_str == 'type of elder abuse':
            return 'type_of_elder_abuse'
        elif value_str == 'type of elder abuse and sex of elderly person being abused':
            return 'type_of_elder_abuse_and_sex'
        elif value_str == "abuser's relationship with elderly person being abused":
            return 'abusers_relationship'
        elif value_str == 'residential district of elderly person being abused':
            return 'residential_district'
        else:
            print(f"Warning: Unexpected value in category: '{value}'")
            return None

    def convert_category_zh(value):
        '''
        Convert category_zh to categorical
        Based on metadata unique_values: ['虐待長者性質', '虐待長者性質及受虐長者性別', '施虐者與受虐長者關係', '受虐長者居住地區']
        Value mapping: {'虐待長者性質': 'elder_abuse_type', '虐待長者性質及受虐長者性別': 'elder_abuse_type_and_sex', '施虐者與受虐長者關係': 'abuser_relationship', '受虐長者居住地區': 'elder_residence'}
        '''
        if pd.isna(value):
            return None
        value_str = str(value).strip().lower()
        value_str = value_str.replace('\u3000', ' ')  # Replace full-width spaces

        if value_str == '虐待長者性質':
            return 'elder_abuse_type'
        elif value_str == '虐待長者性質及受虐長者性別':
            return 'elder_abuse_type_and_sex'
        elif value_str == '施虐者與受虐長者關係':
            return 'abuser_relationship'
        elif value_str == '受虐長者居住地區':
            return 'elder_residence'
        else:
            print(f"Warning: Unexpected value in category_zh: '{value}'")
            return None

    # Apply conversions
    try:
        result['reported_to_police'] = result['Incident Being or Not Reported to Police'].apply(lambda x: True if str(x).strip().lower() == 'yes' else (False if str(x).strip().lower() == 'no' else None))
        print(f'Converted reported_to_police to boolean')
    except Exception as e:
        print(f'Error converting reported_to_police: {e}')

    try:
        result['reported_to_police_zh'] = result['有否向警方舉報事件'].apply(convert_reported_to_police_zh)
        print(f'Converted reported_to_police_zh to categorical')
    except Exception as e:
        print(f'Error converting reported_to_police_zh: {e}')

    try:
        result['category'] = result['Category'].apply(convert_category)
        print(f'Converted category to categorical')
    except Exception as e:
        print(f'Error converting category: {e}')

    try:
        result['category_zh'] = result['類別'].apply(convert_category_zh)
        print(f'Converted category_zh to categorical')
    except Exception as e:
        print(f'Error converting category_zh: {e}')

    try:
        result['year'] = pd.to_numeric(result['年份/Year'], errors='coerce').astype('Int64')
        print(f'Converted year to integer')
    except Exception as e:
        print(f'Error converting year to integer: {e}')

    try:
        result['number_of_cases'] = pd.to_numeric(result['個案數字/No. of Cases'], errors='coerce').astype('Int64')
        print(f'Converted number_of_cases to integer')
    except Exception as e:
        print(f'Error converting number_of_cases to integer: {e}')

    # STEP 3: Rename columns
    rename_map = {
        '年份/Year': 'year',
        '有否向警方舉報事件': 'reported_to_police_zh',
        'Incident Being or Not Reported to Police': 'reported_to_police',
        '類別': 'category_zh',
        'Category': 'category',
        '項目': 'item_zh',
        'Item': 'item',
        '個案數字/No. of Cases': 'number_of_cases'
    }
    result.rename(columns=rename_map, inplace=True)

    # STEP 4: Select and order final columns
    expected_columns = ['year', 'reported_to_police_zh', 'reported_to_police', 'category_zh', 'category', 'item_zh', 'item', 'number_of_cases', 'item_type_zh', 'gender_zh', 'item_type', 'gender']
    
    # Ensure all expected columns exist, filling with None if missing
    for col in expected_columns:
        if col not in result.columns:
            result[col] = None
            print(f"Warning: Column '{col}' not found, filled with None")
    
    result = result[expected_columns]

    return result