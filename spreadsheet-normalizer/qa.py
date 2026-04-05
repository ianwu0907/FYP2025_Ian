from build_qa_dataset import build_qa_dataset, summarise_dataset

# 一次性构建并存到磁盘
dataset = build_qa_dataset(
    raw_dir="raw/",
    tidy_dir="tidy/",
    output_path="dataset/qa_pairs.json",
    n_per_type=2,
)
summarise_dataset(dataset)


