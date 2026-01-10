import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill
import chardet

def detect_encoding(file_path):
    """自动检测文件编码"""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

# 自动检测编码
encoding = detect_encoding("elder.csv")
print(f"检测到的编码: {encoding}")

# 使用检测到的编码读取
df = pd.read_csv("elder.csv", encoding=encoding)

# 保存为Excel
df.to_excel("elder.xlsx", index=False)
print("✅ 转换成功！")