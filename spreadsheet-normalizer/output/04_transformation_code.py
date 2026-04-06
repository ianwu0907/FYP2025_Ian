def transform(df):
    # Step 1: Forward-fill the header rows to reconstruct a multi-level header
    headers = df.iloc[[2, 3]].ffill(axis=1)
    df.columns = headers.iloc[1]  # Take the bottom row for the actual column names

    # Step 2: Drop metadata rows and rows with total/summary data
    records = df.iloc[5:]
    exclusion_list = [
        "所有少數族裔人士[3]", "All ethnic minorities[3]",
        "撇除外籍家庭傭工後的所有少數族裔人士[3]",
        "All ethnic minorities, excluding foreign domestic ...",
        "全港人口", "Whole population", "註釋：", "Notes:"
    ]
    records = records[~records.iloc[:, 0].isin(exclusion_list)].reset_index(drop=True)

    # Step 3: Keep necessary columns (ethnicity, count and percentage for all years)
    records = records.iloc[:, [0, 3, 4, 5, 6, 7, 8]]

    # Step 4: Melt the DataFrame to long format
    records_melted = records.melt(id_vars=records.columns[0], 
                                   value_vars=[records.columns[1], records.columns[2], 
                                               records.columns[3], records.columns[4], 
                                               records.columns[5], records.columns[6]],
                                   var_name='year_type', value_name='value')

    # Step 5: Create mapping for year and type
    year_map = {records.columns[1]: 2011, records.columns[2]: 2011, 
                records.columns[3]: 2016, records.columns[4]: 2016, 
                records.columns[5]: 2021, records.columns[6]: 2021}
    type_map = {records.columns[1]: 'count', records.columns[2]: 'percentage', 
                records.columns[3]: 'count', records.columns[4]: 'percentage', 
                records.columns[5]: 'count', records.columns[6]: 'percentage'}

    records_melted['year'] = records_melted['year_type'].map(year_map)
    records_melted['type'] = records_melted['year_type'].map(type_map)

    # Step 6: Drop the year_type column and pivot to tidy format
    records_melted = records_melted.drop(columns='year_type')
    records_pivot = records_melted.pivot_table(index=[records.columns[0], 'year'], 
                                                columns='type', values='value', 
                                                aggfunc='first').reset_index()

    # Step 7: Rename the columns to match the desired output
    records_pivot.columns = ['ethnicity', 'year', 'count', 'percentage']

    # Convert count and percentage to numeric
    records_pivot['count'] = pd.to_numeric(records_pivot['count'], errors='coerce')
    records_pivot['percentage'] = pd.to_numeric(records_pivot['percentage'], errors='coerce')

    return records_pivot.reset_index(drop=True)