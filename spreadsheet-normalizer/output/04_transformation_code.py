import pandas as pd
import numpy as np

def transform(df):
    print(f"Input shape: {df.shape}")

    # Step 1: Forward fill the year column to handle sparse rows
    df.iloc[:, 0] = df.iloc[:, 0].replace(r'^\s*$', np.nan, regex=True).ffill()
    
    # Step 2: Remove aggregate rows that contain specific keywords
    exclude_labels = ["Total", "Sum", "Average", "身體虐待", "精神虐待", "疏忽照顧", "侵吞財產", "遺棄長者", "性侵犯", "其他"]
    df = df[~df.iloc[:, 5].isin(exclude_labels)]
    print(f"After removing aggregate rows: {df.shape}")
    
    # Step 3: Identify and process the rows based on reporting status
    records = []
    for i in range(len(df)):
        year = df.iloc[i, 0]
        reporting_status = df.iloc[i, 1] if pd.notna(df.iloc[i, 1]) else df.iloc[i, 2]
        abuse_type = df.iloc[i, 5]
        count = df.iloc[i, 7]
        
        # Check if there are valid counts and relevant abuse types
        if pd.notna(reporting_status) and pd.notna(abuse_type) and pd.notna(count):
            # Process gender from abuse_type column
            if "男性" in abuse_type:
                gender = "Male"
            elif "女性" in abuse_type:
                gender = "Female"
            else:
                gender = None
            
            # Split the abuse_type to get main type and gender if applicable
            if '-' in abuse_type:
                main_abuse_type, gender = abuse_type.split(' - ')
                main_abuse_type = main_abuse_type.strip()
            else:
                main_abuse_type = abuse_type
            
            # Append the record if all conditions satisfied
            records.append({
                "year": int(year),
                "reporting_status": reporting_status.strip(),
                "abuse_type": main_abuse_type.strip(),
                "gender": gender,
                "count": int(count)
            })

    # Create DataFrame from collected records
    output = pd.DataFrame(records, columns=['year', 'reporting_status', 'abuse_type', 'gender', 'count'])
    print(f"Final output: {output.shape}")
    return output