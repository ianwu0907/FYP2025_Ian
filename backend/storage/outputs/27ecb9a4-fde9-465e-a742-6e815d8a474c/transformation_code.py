import pandas as pd
import numpy as np

def transform(df):
    """Transform messy spreadsheet to normalized table"""
    result = df.copy()

    # STEP 0: Remove summary rows if detected
    summary_indices = []
    if summary_indices:
        try:
            result = result.drop(index=summary_indices).reset_index(drop=True)
            print(f"Removed {len(summary_indices)} summary rows")
        except Exception as e:
            print(f"Warning: Could not remove summary rows: {e}")

    # STEP 2: Type conversions

    # Define conversion helper functions

    def convert_Incident_Being_or_Not_Reported_to_Police(value):
        '''
        Convert Incident Being or Not Reported to Police to boolean
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
            print(f"Warning: Unexpected value in Incident Being or Not Reported to Police: '{value}'")
            return None

    # Apply conversions
    try:
        result['Incident Being or Not Reported to Police'] = result['Incident Being or Not Reported to Police'].apply(convert_Incident_Being_or_Not_Reported_to_Police)
        print(f'Converted Incident Being or Not Reported to Police to boolean')
    except Exception as e:
        print(f'Error converting Incident Being or Not Reported to Police: {e}')

    # STEP 3: Rename columns
    rename_map = {
        '年份/Year': 'year',
        'Incident Being or Not Reported to Police': 'reported_to_police',
        '類別': 'category_chinese',
        'Category': 'category',
        '項目1': 'item1_chinese',
        '項目2': 'item2_chinese',
        'Item1': 'item1',
        'Item2': 'item2',
        '個案數字/No. of Cases': 'number_of_cases'
    }
    result.rename(columns=rename_map, inplace=True)

    # STEP 4: Select and order final columns
    expected_columns = ['year', 'reported_to_police', 'category_chinese', 'category', 'item1_chinese', 'item2_chinese', 'item1', 'item2', 'number_of_cases']
    result = result[expected_columns]

    try:
        result['year'] = pd.to_numeric(result['year'], errors='coerce').astype('Int64')
        result['number_of_cases'] = pd.to_numeric(result['number_of_cases'], errors='coerce').astype('Int64')
        print('Converted year and number_of_cases to integer')
    except Exception as e:
        print(f'Error converting year and number_of_cases to integer: {e}')

    return result