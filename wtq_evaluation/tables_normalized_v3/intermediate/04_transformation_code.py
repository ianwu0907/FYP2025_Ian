import pandas as pd
import numpy as np
import re

def transform(df):
    data = df.copy()

    def clean_text(x):
        if pd.isna(x):
            return ""
        return str(x).replace("\u3000", " ").replace("\n", " ").strip()

    def to_num(x):
        return pd.to_numeric(x, errors="coerce")

    def parse_year(x):
        y = to_num(x)
        return float(y) if pd.notna(y) else np.nan

    def parse_ethnicity(x):
        s = clean_text(x)
        if not s:
            return None, None

        s = re.sub(r"\s+", " ", s).strip()

        skip_values = {
            "所有少數族裔人士",
            "All ethnic minorities",
            "撇除外籍家庭傭工後的所有少數族裔人士",
            "All ethnic minorities, excluding foreign domestic helpers",
            "全港人口",
            "Total",
            "總計",
            "合計",
            "Overall",
            "少數族裔人士",
            "Ethnic minorities",
        }
        if s in skip_values:
            return None, None

        m = re.match(r"^(.*?)[（(]([^（）()]+)[）)]$", s)
        if m:
            left = clean_text(m.group(1))
            right = clean_text(m.group(2))
            if left and right:
                return left, right

        for pat in [r"^(.*?)\s*[:：]\s*(.+)$", r"^(.*?)\s*[-–—]\s*(.+)$"]:
            m = re.match(pat, s)
            if m:
                left = clean_text(m.group(1))
                right = clean_text(m.group(2))
                if left and right:
                    return left, right

        parts = re.split(r"[、,，/／]", s)
        if len(parts) > 1:
            left = clean_text(parts[0])
            right = clean_text(" / ".join(parts[1:]))
            if left and right:
                return left, right

        return s, None

    column_map = {
        3: ("Never married", "Male"),
        4: ("Never married", "Female"),
        5: ("Never married", "Overall"),
        6: ("Married", "Male"),
        7: ("Married", "Female"),
        8: ("Married", "Overall"),
        9: ("Widowed/divorced/separated", "Male"),
        10: ("Widowed/divorced/separated", "Female"),
        11: ("Widowed/divorced/separated", "Overall"),
        12: ("Total", "Male"),
        13: ("Total", "Female"),
        14: ("Total", "Overall"),
    }

    records = []
    n_rows = data.shape[0]
    n_cols = data.shape[1]

    for i in range(n_rows):
        year = parse_year(data.iloc[i, 0]) if n_cols > 0 else np.nan
        ethnicity, sub_ethnicity = parse_ethnicity(data.iloc[i, 1]) if n_cols > 1 else (None, None)

        if ethnicity is None:
            continue

        for col_idx, (marital_status, sex) in column_map.items():
            if col_idx >= n_cols:
                continue
            val = to_num(data.iloc[i, col_idx])
            if pd.isna(val):
                continue
            records.append(
                {
                    "year": year,
                    "ethnicity": ethnicity,
                    "sub_ethnicity": sub_ethnicity,
                    "marital_status": marital_status,
                    "sex": sex,
                    "percentage": float(val),
                }
            )

    result = pd.DataFrame(
        records,
        columns=["year", "ethnicity", "sub_ethnicity", "marital_status", "sex", "percentage"],
    )

    return result