"""
自动识别本地表格文件在 HiTab QA dataset 中有哪些匹配，
并将匹配到的 QA pair 合并进你的自定义 dataset。

用法（多个 jsonl 用逗号分隔，无空格）：
    python match_and_merge.py \
        --tables_dir  ./hitab_raw_tables/xlsx \
        --custom      ./custom_qa_dataset_updated.json \
        --output      ./merged_output.json \
        --hitab_qa    ./hitab_raw_tables/train_samples.jsonl,./hitab_raw_tables/dev_samples.jsonl,./hitab_raw_tables/test_samples.jsonl

依赖：无额外依赖，纯标准库
"""

import json
import os
import glob
import re
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables_dir", required=True, help="你的表格文件所在目录")
    parser.add_argument("--hitab_qa",   required=True,
                        help="HiTab QA jsonl 文件，多个文件用逗号分隔（无空格），例如: train.jsonl,dev.jsonl,test.jsonl")
    parser.add_argument("--custom",     required=True, help="你的自定义 dataset JSON 文件")
    parser.add_argument("--output",     required=True, help="合并后输出的 JSON 文件路径")
    return parser.parse_args()


def extract_table_id(filename):
    stem = os.path.splitext(filename)[0]
    parts = re.findall(r'\d+', stem)
    for p in parts:
        if len(p) >= 3:
            return p
    return parts[0] if parts else None


def get_answer_type(answer_list):
    for a in answer_list:
        try:
            float(str(a).replace(",", ""))
        except ValueError:
            return "text"
    return "number"


def get_current_max_id(qa_pairs):
    max_num = 0
    for pair in qa_pairs:
        m = re.search(r"(\d+)$", str(pair.get("id", "")))
        if m:
            max_num = max(max_num, int(m.group(1)))
    return max_num


def load_hitab_files(jsonl_paths):
    samples = []
    seen_ids = set()
    for path in jsonl_paths:
        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                sid = sample.get("id", "")
                if sid not in seen_ids:
                    seen_ids.add(sid)
                    samples.append(sample)
                    count += 1
        print(f"   读取 {os.path.basename(path)}: {count} 条")
    return samples


def main():
    args = parse_args()

    # ── 1. 扫描本地文件夹 ──────────────────────────────────────────
    all_files = [f for f in glob.glob(os.path.join(args.tables_dir, "*"))
                 if os.path.isfile(f)]
    print(f"\n📂 本地文件夹: {args.tables_dir}")
    print(f"   共找到 {len(all_files)} 个文件\n")

    id_to_file = {}
    failed = []
    for f in all_files:
        fname = os.path.basename(f)
        tid = extract_table_id(fname)
        if tid:
            id_to_file[tid] = fname
        else:
            failed.append(fname)

    print(f"   成功提取 table_id: {len(id_to_file)} 个")
    if failed:
        print(f"   ⚠️  无法提取 table_id（跳过）: {len(failed)} 个")
        for fn in failed:
            print(f"      {fn}")
    print(f"   table_id 样例: {list(id_to_file.keys())[:10]}\n")

    # ── 2. 读取 HiTab QA（逗号分隔多文件，自动去重）────────────────
    jsonl_paths = [p.strip() for p in args.hitab_qa.split(",")]
    print(f"📋 读取 HiTab QA 文件:")
    hitab_samples = load_hitab_files(jsonl_paths)
    print(f"   合计（去重后）: {len(hitab_samples)} 条\n")

    # ── 3. 匹配 ──────────────────────────────────────────────────
    matched = []
    matched_table_ids = set()
    for sample in hitab_samples:
        tid = str(sample["table_id"])
        if tid in id_to_file:
            matched.append((sample, id_to_file[tid]))
            matched_table_ids.add(tid)

    print(f"✅ 匹配结果:")
    print(f"   命中表格数: {len(matched_table_ids)} 张")
    print(f"   命中 QA pairs: {len(matched)} 条")
    if matched_table_ids:
        print(f"   命中的 table_id: {sorted(matched_table_ids)}")

    if not matched:
        print("\n❌ 没有匹配到任何 QA pair。")
        return

    # ── 4. 加载自定义 dataset ────────────────────────────────────
    with open(args.custom, encoding="utf-8") as f:
        custom = json.load(f)
    existing_pairs = custom["qa_pairs"]
    print(f"\n📦 现有自定义 QA pairs: {len(existing_pairs)} 条")

    # ── 5. 转换并追加 ────────────────────────────────────────────
    max_id = get_current_max_id(existing_pairs)
    new_pairs = []

    for sample, local_filename in matched:
        answers = [str(a) for a in sample.get("answer", [])]
        max_id += 1
        new_pairs.append({
            "id": f"qa-{max_id:03d}",
            "table_file": local_filename,
            "question": sample["question"],
            "answers": answers,
            "answer_type": get_answer_type(answers),
            "source": "hitab",
            "hitab_id": sample.get("id", ""),
            "aggregation": sample.get("aggregation", [])
        })

    merged_pairs = existing_pairs + new_pairs
    custom["qa_pairs"] = merged_pairs
    custom["metadata"]["total_qa_pairs"] = len(merged_pairs)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(custom, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 合并完成:")
    print(f"   新增 {len(new_pairs)} 条 HiTab QA pairs")
    print(f"   总计 {len(merged_pairs)} 条 → {args.output}")


if __name__ == "__main__":
    main()