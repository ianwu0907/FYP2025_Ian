"""
QA Dataset构建工具
帮助从图片和问答对创建标准化的QA dataset
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

class QADatasetBuilder:
    """QA Dataset构建器"""
    
    def __init__(self):
        self.samples = []
        self.dataset_info = {
            "name": "Spreadsheet VLM QA Dataset",
            "version": "1.0",
            "description": "QA dataset for spreadsheet understanding",
            "created_at": datetime.now().isoformat()
        }
    
    def add_sample(self, 
                   image_path: str,
                   question: str,
                   answer: str,
                   sample_id: str = None,
                   question_type: str = "general",
                   difficulty: str = "medium",
                   **metadata):
        """
        添加一个QA样本
        
        Args:
            image_path: 图片路径
            question: 问题
            answer: 标准答案
            sample_id: 样本ID（可选，自动生成）
            question_type: 问题类型 (numerical/categorical/boolean/counting/general)
            difficulty: 难度 (easy/medium/hard)
            **metadata: 其他元数据
        """
        if sample_id is None:
            sample_id = f"sample_{len(self.samples) + 1:03d}"
        
        sample = {
            "id": sample_id,
            "image_path": image_path,
            "question": question,
            "answer": answer,
            "question_type": question_type,
            "difficulty": difficulty
        }
        
        # 添加额外的元数据
        if metadata:
            sample["metadata"] = metadata
        
        self.samples.append(sample)
        print(f"✓ 添加样本 {sample_id}: {question[:50]}...")
    
    def load_from_csv(self, csv_path: str):
        """
        从CSV文件加载QA对
        
        CSV格式:
        image_path,question,answer,question_type,difficulty
        """
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for i, row in enumerate(reader, 1):
                self.add_sample(
                    image_path=row['image_path'],
                    question=row['question'],
                    answer=row['answer'],
                    sample_id=row.get('id', f'sample_{i:03d}'),
                    question_type=row.get('question_type', 'general'),
                    difficulty=row.get('difficulty', 'medium')
                )
        
        print(f"\n✓ 从CSV加载了 {len(self.samples)} 个样本")
    
    def save_json(self, output_path: str):
        """保存为JSON格式"""
        data = {
            "dataset_info": self.dataset_info,
            "samples": self.samples
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 已保存JSON格式: {output_path}")
    
    def save_jsonl(self, output_path: str):
        """保存为JSONL格式"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in self.samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"✓ 已保存JSONL格式: {output_path}")
    
    def save_csv(self, output_path: str):
        """保存为CSV格式"""
        if not self.samples:
            print("❌ 没有样本可保存")
            return
        
        # 获取所有字段
        fieldnames = ['id', 'image_path', 'question', 'answer', 
                     'question_type', 'difficulty']
        
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for sample in self.samples:
                row = {k: sample.get(k, '') for k in fieldnames}
                writer.writerow(row)
        
        print(f"✓ 已保存CSV格式: {output_path}")
    
    def print_summary(self):
        """打印dataset摘要"""
        print("\n" + "=" * 60)
        print("📊 Dataset摘要")
        print("=" * 60)
        print(f"总样本数: {len(self.samples)}")
        
        # 按类型统计
        type_counts = {}
        for sample in self.samples:
            qtype = sample['question_type']
            type_counts[qtype] = type_counts.get(qtype, 0) + 1
        
        print("\n按问题类型:")
        for qtype, count in sorted(type_counts.items()):
            print(f"  {qtype}: {count}")
        
        # 按难度统计
        diff_counts = {}
        for sample in self.samples:
            diff = sample.get('difficulty', 'unknown')
            diff_counts[diff] = diff_counts.get(diff, 0) + 1
        
        print("\n按难度:")
        for diff, count in sorted(diff_counts.items()):
            print(f"  {diff}: {count}")


def create_template_csv(output_path: str = 'qa_template.csv'):
    """创建CSV模板"""
    template = [
        {
            'id': 'sample_001',
            'image_path': 'picture/file1.png',
            'question': 'What is the total revenue?',
            'answer': '125000',
            'question_type': 'numerical',
            'difficulty': 'easy'
        },
        {
            'id': 'sample_002',
            'image_path': 'picture/file2.png',
            'question': 'Which product has the highest sales?',
            'answer': 'Product A',
            'question_type': 'categorical',
            'difficulty': 'medium'
        },
        {
            'id': 'sample_003',
            'image_path': 'picture/file3.png',
            'question': 'Is the total greater than 1000?',
            'answer': 'yes',
            'question_type': 'boolean',
            'difficulty': 'easy'
        }
    ]
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['id', 'image_path', 'question', 'answer', 
                     'question_type', 'difficulty']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(template)
    
    print(f"✓ CSV模板已创建: {output_path}")
    print("\n使用方法:")
    print("1. 在Excel中打开此文件")
    print("2. 填写你的问答对")
    print("3. 保存")
    print("4. 运行: python build_dataset.py --from-csv qa_template.csv")


def interactive_build():
    """交互式构建dataset"""
    print("\n" + "=" * 60)
    print("🔧 交互式Dataset构建")
    print("=" * 60)
    
    builder = QADatasetBuilder()
    
    # 获取dataset信息
    builder.dataset_info['name'] = input("\nDataset名称 [默认: Spreadsheet VLM QA Dataset]: ").strip() or "Spreadsheet VLM QA Dataset"
    builder.dataset_info['description'] = input("Dataset描述 [默认: QA dataset]: ").strip() or "QA dataset for spreadsheet understanding"
    
    print("\n开始添加样本（输入空的image_path退出）:")
    
    sample_num = 1
    while True:
        print(f"\n--- 样本 {sample_num} ---")
        
        image_path = input("图片路径: ").strip()
        if not image_path:
            break
        
        question = input("问题: ").strip()
        if not question:
            print("⚠️  问题不能为空，跳过此样本")
            continue
        
        answer = input("答案: ").strip()
        if not answer:
            print("⚠️  答案不能为空，跳过此样本")
            continue
        
        question_type = input("问题类型 [numerical/categorical/boolean/counting/general, 默认: general]: ").strip() or "general"
        difficulty = input("难度 [easy/medium/hard, 默认: medium]: ").strip() or "medium"
        
        builder.add_sample(
            image_path=image_path,
            question=question,
            answer=answer,
            question_type=question_type,
            difficulty=difficulty
        )
        
        sample_num += 1
    
    if len(builder.samples) == 0:
        print("\n❌ 没有添加任何样本")
        return
    
    # 打印摘要
    builder.print_summary()
    
    # 保存
    print("\n保存格式:")
    print("1. JSON")
    print("2. JSONL")
    print("3. CSV")
    print("4. 全部")
    
    choice = input("\n选择格式 [1-4, 默认: 1]: ").strip() or "1"
    output_name = input("输出文件名前缀 [默认: qa_dataset]: ").strip() or "qa_dataset"
    
    if choice in ['1', '4']:
        builder.save_json(f"{output_name}.json")
    if choice in ['2', '4']:
        builder.save_jsonl(f"{output_name}.jsonl")
    if choice in ['3', '4']:
        builder.save_csv(f"{output_name}.csv")
    
    print("\n✅ Dataset创建完成!")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='QA Dataset构建工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 交互式创建
  python build_dataset.py --interactive
  
  # 创建CSV模板
  python build_dataset.py --create-template
  
  # 从CSV构建
  python build_dataset.py --from-csv qa_data.csv -o qa_dataset.json
  
  # 从CSV构建并保存为多种格式
  python build_dataset.py --from-csv qa_data.csv --all-formats
        """
    )
    
    parser.add_argument('--interactive', action='store_true',
                       help='交互式创建dataset')
    parser.add_argument('--create-template', action='store_true',
                       help='创建CSV模板')
    parser.add_argument('--from-csv', help='从CSV文件构建')
    parser.add_argument('-o', '--output', default='qa_dataset',
                       help='输出文件名（不含扩展名）')
    parser.add_argument('--format', choices=['json', 'jsonl', 'csv'],
                       default='json',
                       help='输出格式')
    parser.add_argument('--all-formats', action='store_true',
                       help='保存为所有格式')
    
    args = parser.parse_args()
    
    # 创建模板
    if args.create_template:
        create_template_csv()
        return
    
    # 交互式创建
    if args.interactive:
        interactive_build()
        return
    
    # 从CSV构建
    if args.from_csv:
        builder = QADatasetBuilder()
        builder.load_from_csv(args.from_csv)
        builder.print_summary()
        
        if args.all_formats:
            builder.save_json(f"{args.output}.json")
            builder.save_jsonl(f"{args.output}.jsonl")
            builder.save_csv(f"{args.output}.csv")
        else:
            if args.format == 'json':
                builder.save_json(f"{args.output}.json")
            elif args.format == 'jsonl':
                builder.save_jsonl(f"{args.output}.jsonl")
            elif args.format == 'csv':
                builder.save_csv(f"{args.output}.csv")
        
        return
    
    # 如果没有任何参数，显示帮助
    parser.print_help()


if __name__ == '__main__':
    main()