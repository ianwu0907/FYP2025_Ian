def transform(df):
    # Multi-level header processing
    headers = df.iloc[[0, 1]].ffill(axis=1)
    print(headers)

    # Prepare the final DataFrame
    result = df.copy()
    
    # Extract region of birth
    result['region_of_birth'] = result.iloc[:, 0]
    print(result)

    # Define the education types and their respective columns
    education_types = [
        ("non-university or university postsecondary diploma", 1),
        ("university degree only", 2),
        ("non-university or university postsecondary diploma", 3),
        ("university degree only", 4)
    ]

    # Initialize an empty list to gather rows for the final DataFrame
    tidy_data = []

    # Iterate through the original DataFrame
    for i in range(2, len(result)):
        if i < 2 or result.iloc[i, 0] == "country of birth":
            continue
        
        for edu_type, col_idx in education_types:
            tidy_data.append({
                'region_of_birth': result.iloc[i, 0],
                'education_type': edu_type,
                'percent': result.iloc[i, col_idx]
            })

    tidy_df = pd.DataFrame(tidy_data)
    print(tidy_df)

    return tidy_df.reset_index(drop=True)