import pandas as pd
import numpy as np

def transform(df):
    """
    Transform messy spreadsheet to tidy format.
    """
    try:
        print("Step 0: Starting transformation")
        print("Initial DataFrame shape:", df.shape)
        
        # === Step 1: Extract the core data region ===
        # Remove header rows (0 to 5); rows 6 onward are the core data.
        df_core = df.iloc[6:].copy()
        print("After extracting core data region, shape:", df_core.shape)
        
        # === Step 2: Identify and assign section markers for year ===
        # In column 0, non-empty cells indicate a new year, so replace empty strings with NaN and forward fill.
        df_core['year'] = df_core.iloc[:,0].replace('', np.nan).ffill()
        print("After assigning year markers, sample 'year' values:")
        print(df_core['year'].head(10))
        
        # === Step 3: Pair bilingual rows for each ethnic group ===
        # According to strategy, row at index 0 of df_core is a section marker row; the following rows come in pairs.
        # Drop the first row (the marker row), then pair subsequent rows (CN row followed by EN row).
        df_records = df_core.iloc[1:].reset_index(drop=True).copy()
        # Create a pairing id: every 2 consecutive records form one record.
        df_records['pair_id'] = df_records.index // 2
        
        # Select Chinese (CN) rows (first row in each pair) and then English (EN) rows (second row in each pair)
        df_cn = df_records.iloc[::2].reset_index(drop=True).copy()
        df_en = df_records.iloc[1::2].reset_index(drop=True).copy()
        
        print("After pairing rows, number of pairs:", len(df_cn))
        print("Sample CN rows (first 3 rows, first 5 columns):")
        print(df_cn.iloc[:3, :5])
        print("Sample EN rows (first 3 rows, first 5 columns):")
        print(df_en.iloc[:3, :5])
        
        # Propagate pair_id explicitly (even though already there)
        df_cn['pair_id'] = df_cn.index
        df_en['pair_id'] = df_en.index
        
        # Rename columns for clarity.
        # For CN rows, assume:
        # - Column 1 holds the Chinese ethnicity name.
        # - Column 11 holds the median_age.
        df_cn = df_cn.rename(columns={ df_cn.columns[1]: 'ethnicity_cn', df_cn.columns[11]: 'median_age' })
        # For EN rows, column 1 holds the English ethnicity name.
        df_en = df_en.rename(columns={ df_en.columns[1]: 'ethnicity_en' })
        
        print("After renaming, sample Chinese ethnicity names:")
        print(df_cn['ethnicity_cn'].head(5))
        print("After renaming, sample English ethnicity names:")
        print(df_en['ethnicity_en'].head(5))
        
        # === Step 4: Unpivot age-specific columns ===
        # The age group headers should be taken from header row 3 (0-indexed row 3 in the original DataFrame)
        # Extract labels from columns 3 to 9
        age_group_labels = [str(x).strip() for x in df.iloc[3, 3:10].tolist()]
        print("Extracted age group labels from original row 3:", age_group_labels)
        
        # Determine the age columns from the CN rows using their column position.
        # We assume the count columns for age groups are in positions 3 to 9.
        age_cols = df_cn.columns[3:10]
        print("Age columns from CN DataFrame (by position):", list(age_cols))
        
        # Build a mapping dict from the column name to the age group label.
        mapping_dict = dict(zip(age_cols, age_group_labels))
        print("Mapping of age column to age group label:", mapping_dict)
        
        # Melt the count values from the Chinese rows.
        df_cn_melt = pd.melt(df_cn,
                             id_vars=['pair_id', 'year', 'ethnicity_cn', 'median_age'],
                             value_vars=age_cols,
                             var_name='age_col',
                             value_name='count')
        # Melt the percentage values from the English rows.
        df_en_melt = pd.melt(df_en,
                             id_vars=['pair_id', 'year', 'ethnicity_en'],
                             value_vars=age_cols,
                             var_name='age_col',
                             value_name='percentage')
        
        # Merge the two melted DataFrames on pair_id, year, and age_col.
        df_merged = pd.merge(df_cn_melt, df_en_melt, on=['pair_id', 'year', 'age_col'])
        
        # Map the 'age_col' using our mapping dictionary to get the actual age_group labels.
        df_merged['age_group'] = df_merged['age_col'].map(mapping_dict)
        # Drop the temporary columns.
        df_merged.drop(columns=['age_col', 'pair_id'], inplace=True)
        
        print("After unpivoting age columns, sample data:")
        print(df_merged.head(10))
        
        # === Step 5: Perform data cleaning and type conversion ===
        # Trim whitespace from text fields.
        df_merged['ethnicity_cn'] = df_merged['ethnicity_cn'].astype(str).str.strip()
        df_merged['ethnicity_en'] = df_merged['ethnicity_en'].astype(str).str.strip()
        df_merged['age_group']    = df_merged['age_group'].astype(str).str.strip()
        
        # Convert 'year' to integer, handling conversion issues.
        df_merged['year'] = pd.to_numeric(df_merged['year'], errors='coerce').astype('Int64')
        
        # Clean and convert count values to integers.
        df_merged['count'] = pd.to_numeric(
            df_merged['count'].astype(str).str.strip(), errors='coerce'
        ).fillna(0).astype(int)
        
        # Remove '%' from percentage values and convert to float.
        df_merged['percentage'] = (
            df_merged['percentage']
            .astype(str)
            .str.replace('%', '', regex=False)
            .str.strip()
        )
        df_merged['percentage'] = pd.to_numeric(
            df_merged['percentage'], errors='coerce'
        ).fillna(0.0).astype(float)
        
        # Convert median_age to float.
        df_merged['median_age'] = pd.to_numeric(
            df_merged['median_age'].astype(str).str.strip(), errors='coerce'
        ).fillna(0.0).astype(float)
        
        print("After data cleaning and type conversion, sample data:")
        print(df_merged.head(10))
        
        # === (Optional) Step 5.5: Filter out aggregate rows if they exist ===
        # In our debug, the expected record is for a specific ethnicity ("亞洲人（非華人）")
        # Sometimes an aggregate row like "All ethnic minorities, excluding foreign domestic helpers" appears.
        # Remove such rows by filtering out the aggregate text if present.
        if not df_merged.empty:
            before_filter = df_merged.shape[0]
            df_merged = df_merged.loc[df_merged['ethnicity_cn'] != "All ethnic minorities, excluding foreign domestic helpers"]
            after_filter = df_merged.shape[0]
            print(f"Filtered out aggregate row(s): {before_filter - after_filter} row(s) removed.")
        
        # === Step 6: Finalize column selection and reordering ===
        # Select the target columns in the desired order.
        result = df_merged[['year', 'ethnicity_cn', 'ethnicity_en', 'age_group', 'count', 'percentage', 'median_age']].copy()
        # Sort the DataFrame by year, ethnicity_cn, then age_group.
        result.sort_values(by=['year', 'ethnicity_cn', 'age_group'], inplace=True)
        result.reset_index(drop=True, inplace=True)
        
        print("Final transformed DataFrame shape:", result.shape)
        print("Final transformed DataFrame sample:")
        print(result.head(10))
        
        return result
    except Exception as e:
        print("Error during transformation:", e)
        return pd.DataFrame()