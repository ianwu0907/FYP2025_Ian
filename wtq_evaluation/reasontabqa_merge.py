"""
从 ReasonTabQA 的 jsonl QA 文件中，匹配 all_tables 里已有的表格，
将对应 QA pair 合并进你的自定义 dataset。

用法：
    python reasontabqa_merge.py \
        --tables_dir  ./all_tables \
        --qa_file     ./data/en/en_20percent_test.jsonl \
        --custom      ./merged_output_wide.json \
        --output      ./merged_output_reasontab.json

可选：同时传 train 和 test（逗号分隔）：
    --qa_file ./data/en/en_20percent_test.jsonl,./data/en/en_80percent_train.jsonl

依赖：无额外依赖，纯标准库
"""

import json
import os
import re
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables_dir", required=True, help="all_tables 目录路径")
    parser.add_argument("--qa_file",    required=True,
                        help="ReasonTabQA jsonl 文件，多个用逗号分隔")
    parser.add_argument("--custom",     required=True, help="你的自定义 dataset JSON 文件")
    parser.add_argument("--output",     required=True, help="合并后输出的 JSON 文件路径")
    parser.add_argument("--one_per_table", action="store_true",
                        help="每张表只保留一条 QA pair（默认关闭）")
    return parser.parse_args()


def get_current_max_id(qa_pairs):
    max_num = 0
    for pair in qa_pairs:
        m = re.search(r"(\d+)$", str(pair.get("id", "")))
        if m:
            max_num = max(max_num, int(m.group(1)))
    return max_num


def get_answer_type(val):
    try:
        float(str(val).replace(",", ""))
        return "number"
    except ValueError:
        return "text"


def extract_table_filename(spreedsheetpath_list):
    """
    从 spreedsheetpath_list 提取文件名。
    格式示例：["Zhongling_Market/.../Table.xls"]
    取第一个路径的文件名部分。
    """
    if not spreedsheetpath_list:
        return None
    path = spreedsheetpath_list[0]
    return os.path.basename(path)


def load_qa_files(qa_paths):
    samples = []
    seen_ids = set()
    for path in qa_paths:
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

    # ── 1. 扫描 all_tables 目录 ───────────────────────────────────
    local_files = set(os.listdir(args.tables_dir))
    print(f"\n📂 本地表格文件: {len(local_files)} 个\n")

    # ── 2. 读取 QA 文件 ──────────────────────────────────────────
    qa_paths = [p.strip() for p in args.qa_file.split(",")]
    print(f"📋 读取 ReasonTabQA QA 文件:")
    samples = load_qa_files(qa_paths)
    print(f"   合计（去重后）: {len(samples)} 条\n")

    # ── 3. 匹配 ──────────────────────────────────────────────────
    matched = []
    matched_tables = set()

    for sample in samples:
        path_list = sample.get("spreedsheetpath_list", [])
        fname = extract_table_filename(path_list)
        if fname and fname in local_files:
            matched.append((sample, fname))
            matched_tables.add(fname)

    print(f"✅ 匹配结果:")
    print(f"   命中表格数: {len(matched_tables)} 张")
    print(f"   命中 QA pairs: {len(matched)} 条")

    if not matched:
        print("\n❌ 没有匹配到任何 QA pair。")
        print("   请检查 spreedsheetpath_list 里的文件名是否与 all_tables 目录一致。")
        print(f"   QA 文件名示例: {extract_table_filename(samples[0].get('spreedsheetpath_list', []))}")
        print(f"   本地文件名示例: {list(local_files)[:3]}")
        return

    # 每张表只保留一条（可选）
    if args.one_per_table:
        seen = set()
        filtered = []
        for item in matched:
            if item[1] not in seen:
                seen.add(item[1])
                filtered.append(item)
        print(f"   （每表1条过滤后）: {len(filtered)} 条")
        matched = filtered

    # ── 4. 加载自定义 dataset ────────────────────────────────────
    with open(args.custom, encoding="utf-8") as f:
        custom = json.load(f)
    existing_pairs = custom["qa_pairs"]
    max_id = get_current_max_id(existing_pairs)
    print(f"\n📦 现有 QA pairs: {len(existing_pairs)} 条")

    # ── 5. 转换并追加 ────────────────────────────────────────────
    new_pairs = []
    for sample, fname in matched:
        answer = str(sample.get("gold_truth", "")).strip()
        max_id += 1
        new_pairs.append({
            "id": f"qa-{max_id:03d}",
            "table_file": fname,
            "question": sample.get("question", "").strip(),
            "answers": [answer],
            "answer_type": get_answer_type(answer),
            "source": "reasontabqa",
            "reasontabqa_id": str(sample.get("id", "")),
            "table_difficulty": sample.get("table_difficulty", ""),
            "question_difficulty": sample.get("question_difficulty", ""),
            "question_type": sample.get("question_type", ""),
            "if_complex_table_header": sample.get("if_complex_table_header", "")
        })

    merged = existing_pairs + new_pairs
    custom["qa_pairs"] = merged
    custom["metadata"]["total_qa_pairs"] = len(merged)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(custom, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 合并完成:")
    print(f"   新增 {len(new_pairs)} 条 ReasonTabQA QA pairs")
    print(f"   总计 {len(merged)} 条 → {args.output}")


if __name__ == "__main__":
    main()
