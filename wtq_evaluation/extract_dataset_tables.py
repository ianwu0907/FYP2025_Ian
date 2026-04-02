"""
把 QA dataset 里涉及的表格从 table_original 复制到 tables_for_dataset。

用法：
    python extract_dataset_tables.py \
        --custom    ./merged_output_cleaned.json \
        --src_dir   ./table_original \
        --dst_dir   ./tables_for_dataset
"""

import json
import os
import shutil
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom",  required=True, help="QA dataset JSON 文件")
    parser.add_argument("--src_dir", required=True, help="所有表格所在目录")
    parser.add_argument("--dst_dir", required=True, help="输出目录")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.dst_dir, exist_ok=True)

    with open(args.custom, encoding="utf-8") as f:
        data = json.load(f)

    needed = set(os.path.basename(p["table_file"]) for p in data["qa_pairs"])
    print(f"\n📋 dataset 涉及表格: {len(needed)} 张")
    print(f"📂 来源目录: {args.src_dir}")
    print(f"📂 输出目录: {args.dst_dir}\n")

    found, missing = [], []
    for fname in sorted(needed):
        src = os.path.join(args.src_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.dst_dir, fname))
            found.append(fname)
        else:
            missing.append(fname)

    print(f"✅ 复制成功: {len(found)} 张")
    if missing:
        print(f"⚠️  找不到: {len(missing)} 张")
        for f in missing:
            print(f"   {f}")
    print(f"\n🎉 完成 → {args.dst_dir}")


if __name__ == "__main__":
    main()
