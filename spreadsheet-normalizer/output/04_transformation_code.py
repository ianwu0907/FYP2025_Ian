def transform(df):
    import pandas as pd
    import numpy as np

    data = df.copy()

    def clean_text(s):
        return (
            s.astype("string")
            .replace(r"^\s*$", pd.NA, regex=True)
            .str.replace("\u3000", " ", regex=False)
            .str.replace("\xa0", " ", regex=False)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    year = pd.to_numeric(data.iloc[:, 0], errors="coerce").ffill()
    report_to_police = clean_text(data.iloc[:, 1]).ffill()

    col2 = clean_text(data.iloc[:, 2])
    group = clean_text(data.iloc[:, 3])
    col4 = clean_text(data.iloc[:, 4])
    item = clean_text(data.iloc[:, 5])
    col6 = clean_text(data.iloc[:, 6])
    cases = pd.to_numeric(data.iloc[:, 7], errors="coerce")

    rel_map = {
        "子": "Son",
        "女": "Daughter",
        "女婿": "Son-in-law",
        "媳婦": "Daughter-in-law",
        "配偶／親密伴侶": "Spouse/ intimate partner",
        "配偶/親密伴侶": "Spouse/ intimate partner",
        "配偶／ 親密伴侶": "Spouse/ intimate partner",
        "孫／外孫": "Grandchildren",
        "親戚": "Relative",
        "朋友／鄰居": "Friend/ neighbour",
        "沒有親戚關係但同住": "Not relative but living together",
        "家庭傭工": "Domestic helper",
        "提供服務給長者的機構員工": "Staff of the agency providing services for the elderly",
        "其他": "Others",
        "中西區": "Central and Western",
        "南區": "Southern",
        "離島區": "Islands",
        "東區": "Eastern",
        "灣仔區": "Wan Chai",
        "九龍城區": "Kowloon City",
        "油尖旺區": "Yau Tsim Mong",
        "深水埗區": "Sham Shui Po",
        "黃大仙區": "Wong Tai Sin",
        "西貢區": "Sai Kung",
        "觀塘區": "Kwun Tong",
        "沙田區": "Sha Tin",
        "大埔區": "Tai Po",
        "北區": "North",
        "元朗區": "Yuen Long",
        "荃灣區": "Tsuen Wan",
        "葵青區": "Kwai Tsing",
        "屯門區": "Tuen Mun",
        "不知道": "Unknown",
        "不適用": "Not applicable",
    }

    def normalize_rel(v):
        if pd.isna(v):
            return pd.NA
        s = str(v).strip()
        if s in rel_map:
            return rel_map[s]
        s2 = s.replace("／", "/").replace(" ", " ")
        s2 = " ".join(s2.split())
        return rel_map.get(s2, s)

    def split_abuse_gender(v):
        if pd.isna(v):
            return pd.Series([pd.NA, pd.NA])
        s = str(v).strip()
        if " - " in s:
            left, right = s.split(" - ", 1)
            return pd.Series([left.strip(), right.strip()])
        if "-" in s:
            left, right = s.split("-", 1)
            return pd.Series([left.strip(), right.strip()])
        return pd.Series([s, pd.NA])

    split_item = item.apply(split_abuse_gender)
    abuse_type = split_item.iloc[:, 0].replace({"Total": pd.NA, "多種虐待": pd.NA, "未適用": pd.NA})
    gender = split_item.iloc[:, 1]

    def is_rel_like(v):
        if pd.isna(v):
            return False
        s = str(v).strip()
        if s in rel_map:
            return True
        s2 = s.replace("／", "/").replace(" ", " ")
        s2 = " ".join(s2.split())
        return s2 in rel_map

    rel_source = pd.Series(pd.NA, index=data.index, dtype="object")
    rel_source = rel_source.where(~group.apply(is_rel_like), group)
    rel_source = rel_source.where(~col2.apply(is_rel_like), col2)
    rel_source = rel_source.where(~col4.apply(is_rel_like), col4)
    rel_source = rel_source.where(~item.apply(is_rel_like), item)
    rel_source = rel_source.where(~col6.apply(is_rel_like), col6)

    relationship_to_elder = rel_source.map(normalize_rel)

    mask_table = group.notna() & (
        group.str.contains("虐待長者性質及受虐長者性別", na=False)
        | group.str.contains("Type of Elder Abuse", na=False)
        | group.str.contains("受虐長者", na=False)
        | group.str.contains("Sex of Elderly", na=False)
    )

    out = pd.DataFrame(
        {
            "year": pd.to_numeric(year, errors="coerce"),
            "report_to_police": report_to_police,
            "abuse_type": abuse_type,
            "gender": gender,
            "relationship_to_elder": relationship_to_elder,
            "no_of_cases": cases,
            "_group": group,
            "_item": item,
        }
    )

    out = out.loc[mask_table].copy()
    out = out.loc[out["abuse_type"].notna()].copy()
    out = out.loc[~out["abuse_type"].isin(["Total", "多種虐待", "未適用"])].copy()
    out = out.loc[out["no_of_cases"].notna()].copy()

    out["relationship_to_elder"] = out["relationship_to_elder"].where(
        out["relationship_to_elder"].notna(),
        out["_group"].map(normalize_rel),
    )
    out["relationship_to_elder"] = out["relationship_to_elder"].where(
        out["relationship_to_elder"].notna(),
        out["_item"].map(normalize_rel),
    )
    out["relationship_to_elder"] = out["relationship_to_elder"].where(
        out["relationship_to_elder"].notna(),
        col2.loc[out.index].map(normalize_rel),
    )
    out["relationship_to_elder"] = out["relationship_to_elder"].where(
        out["relationship_to_elder"].notna(),
        col4.loc[out.index].map(normalize_rel),
    )

    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["no_of_cases"] = pd.to_numeric(out["no_of_cases"], errors="coerce").astype("Int64")

    out = out.loc[out["no_of_cases"].notna()].copy()

    out = out[
        [
            "year",
            "report_to_police",
            "abuse_type",
            "gender",
            "relationship_to_elder",
            "no_of_cases",
        ]
    ].copy()

    return out