import json
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import re
from difflib import SequenceMatcher

# ============================================
# 评估器类
# ============================================

class SpreadsheetEvaluator:
    """电子表格理解任务评估器"""
    
    def __init__(self, qa_file: str):
        with open(qa_file, 'r', encoding='utf-8') as f:
            self.qa_data = json.load(f)
        self.qa_dict = {qa['id']: qa for qa in self.qa_data['qa_pairs']}
    
    def normalize_text(self, text: str) -> str:
        """文本标准化"""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s:,]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def parse_cell_range(self, text: str) -> Dict:
        """解析单元格范围"""
        text_upper = text.upper()
        result = {'has_range': False, 'top_left': None, 'bottom_right': None}
        
        # Pattern: "A19:F54"
        pattern = r'([A-Z]+\d+)\s*:\s*([A-Z]+\d+)'
        match = re.search(pattern, text_upper)
        
        if match:
            result['has_range'] = True
            result['top_left'] = match.group(1)
            result['bottom_right'] = match.group(2)
        
        return result
    
    def evaluate_range(self, predicted: str, answer: str) -> float:
        """评估range类型答案"""
        pred_range = self.parse_cell_range(predicted)
        ans_range = self.parse_cell_range(answer)
        
        if not pred_range['has_range']:
            return 0.0
        
        # 精确匹配
        if (pred_range['top_left'] == ans_range['top_left'] and 
            pred_range['bottom_right'] == ans_range['bottom_right']):
            return 1.0
        
        # 部分正确
        if pred_range['top_left'] == ans_range['top_left']:
            return 0.5
        if pred_range['bottom_right'] == ans_range['bottom_right']:
            return 0.3
        
        return 0.0
    
    def evaluate_list(self, predicted: str, answer: List[str]) -> float:
        """评估list类型答案"""
        pred_lower = predicted.lower()
        correct = sum(1 for item in answer if item.lower() in pred_lower)
        return correct / len(answer)
    
    def evaluate_single_value(self, predicted: str, answer: str, alternatives: List[str] = None) -> float:
        """评估单值答案"""
        pred_norm = self.normalize_text(predicted)
        ans_norm = self.normalize_text(answer)
        
        # 精确匹配
        if ans_norm in pred_norm or pred_norm in ans_norm:
            return 1.0
        
        # 检查备选答案
        if alternatives:
            for alt in alternatives:
                if self.normalize_text(alt) in pred_norm:
                    return 1.0
        
        # 模糊匹配
        similarity = SequenceMatcher(None, pred_norm, ans_norm).ratio()
        return similarity if similarity >= 0.8 else 0.0
    
    def evaluate_keywords(self, predicted: str, keywords: List[str], min_keywords: int = 1) -> float:
        """评估关键词匹配"""
        pred_lower = predicted.lower()
        matched = sum(1 for kw in keywords if kw.lower() in pred_lower)
        return matched / len(keywords) if matched >= min_keywords else 0.0
    
    def evaluate_single_qa(self, qa_id: str, vlm_answer: str) -> Dict:
        """评估单个QA对"""
        qa_item = self.qa_dict.get(qa_id)
        if not qa_item:
            return {'score': 0.0, 'error': 'QA not found'}
        
        # 检查错误
        if vlm_answer.startswith("ERROR"):
            return {
                'qa_id': qa_id,
                'category': qa_item['category'],
                'difficulty': qa_item.get('difficulty', 'medium'),
                'score': 0.0,
                'error': True,
                'expected': qa_item['answer'],
                'predicted': vlm_answer
            }
        
        answer_type = qa_item.get('answer_type', 'single_value')
        score = 0.0
        
        # 根据类型评估
        if answer_type in ['range', 'cell_range']:
            score = self.evaluate_range(vlm_answer, qa_item['answer'])
        elif answer_type == 'list':
            score = self.evaluate_list(vlm_answer, qa_item['answer'])
        elif answer_type in ['single_value', 'numeric', 'cell_address']:
            score = self.evaluate_single_value(
                vlm_answer, 
                qa_item['answer'],
                qa_item.get('alternative_answers')
            )
        elif answer_type == 'semantic_reasoning':
            score = self.evaluate_keywords(
                vlm_answer,
                qa_item.get('answer_keywords', []),
                qa_item.get('min_keywords', 1)
            )
        
        return {
            'qa_id': qa_id,
            'category': qa_item['category'],
            'difficulty': qa_item.get('difficulty', 'medium'),
            'score': score,
            'expected': str(qa_item['answer'])[:100],
            'predicted': vlm_answer[:100]
        }
    
    def evaluate_all(self, responses: Dict[str, str]) -> Dict:
        """评估所有回答"""
        results = []
        
        for qa_id, answer in responses.items():
            result = self.evaluate_single_qa(qa_id, answer)
            results.append(result)
        
        return self.generate_summary(results)
    
    def generate_summary(self, results: List[Dict]) -> Dict:
        """生成评估摘要"""
        total = len(results)
        total_score = sum(r['score'] for r in results)
        
        # 按类别统计
        by_category = {}
        for r in results:
            cat = r['category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(r)
        
        category_scores = {
            cat: sum(r['score'] for r in rs) / len(rs)
            for cat, rs in by_category.items()
        }
        
        # 按难度统计
        by_difficulty = {}
        for r in results:
            diff = r.get('difficulty', 'medium')
            if diff not in by_difficulty:
                by_difficulty[diff] = []
            by_difficulty[diff].append(r)
        
        difficulty_scores = {
            diff: sum(r['score'] for r in rs) / len(rs)
            for diff, rs in by_difficulty.items()
        }
        
        # 错误统计
        errors = [r for r in results if r.get('error', False)]
        low_scores = [r for r in results if r['score'] < 0.5 and not r.get('error')]
        
        return {
            'overall': {
                'total_questions': total,
                'average_score': total_score / total if total > 0 else 0,
                'percentage': (total_score / total) * 100 if total > 0 else 0
            },
            'by_category': category_scores,
            'by_difficulty': difficulty_scores,
            'errors': errors,
            'low_scores': low_scores,
            'detailed_results': results
        }

# ============================================
# 主评估函数
# ============================================

def evaluate_model_version(model_name: str, version: str, evaluator: SpreadsheetEvaluator, save_report: bool = True):
    """评估指定模型的特定版本"""
    
    model_dir = Path("outputs") / model_name / version
    
    if not model_dir.exists():
        print(f"❌ Directory not found: {model_dir}")
        return None
    
    # 查找最新的响应文件
    response_files = list(model_dir.glob("responses_*.json"))
    if not response_files:
        print(f"❌ No response files found in {model_dir}")
        return None
    
    latest_response = max(response_files, key=lambda p: p.stat().st_mtime)
    
    # 加载响应
    with open(latest_response, 'r', encoding='utf-8') as f:
        responses = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"📊 Evaluating: {model_name} - {version}")
    print(f"{'='*70}")
    print(f"📂 File: {latest_response.name}")
    
    summary = evaluator.evaluate_all(responses)
    
    # 显示结果
    print(f"\n🎯 Overall Performance:")
    print(f"   Total Questions: {summary['overall']['total_questions']}")
    print(f"   Average Score: {summary['overall']['average_score']:.3f}")
    print(f"   Percentage: {summary['overall']['percentage']:.1f}%")
    
    print(f"\n📑 By Category:")
    for cat, score in sorted(summary['by_category'].items()):
        print(f"   {cat:30s}: {score*100:5.1f}%")
    
    print(f"\n⭐ By Difficulty:")
    for diff, score in sorted(summary['by_difficulty'].items()):
        print(f"   {diff:10s}: {score*100:5.1f}%")
    
    if summary['errors']:
        print(f"\n❌ Errors ({len(summary['errors'])}):")
        for err in summary['errors'][:5]:
            print(f"   - {err['qa_id']}")
    
    if summary['low_scores']:
        print(f"\n⚠️  Low Scores (<50%, {len(summary['low_scores'])}):")
        for low in summary['low_scores'][:5]:
            print(f"   - {low['qa_id']}: {low['score']:.2f}")
            if len(low['predicted']) > 0:
                print(f"     Expected: {low['expected'][:60]}...")
                print(f"     Got: {low['predicted'][:60]}...")
    
    # 保存详细报告
    if save_report:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = model_dir / f"evaluation_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Detailed report saved: {report_file.name}")
    
    return summary


def compare_versions(model_name: str, evaluator: SpreadsheetEvaluator):
    """对比同一模型的vanilla和formatted版本"""
    
    print(f"\n{'='*70}")
    print(f"📊 Comparing Versions: {model_name}")
    print(f"{'='*70}")
    
    vanilla_summary = evaluate_model_version(model_name, 'vanilla', evaluator, save_report=False)
    formatted_summary = evaluate_model_version(model_name, 'formatted', evaluator, save_report=False)
    
    if not vanilla_summary or not formatted_summary:
        print("\n❌ Cannot compare - missing data")
        return
    
    # 生成对比表
    print(f"\n{'='*70}")
    print(f"📈 Version Comparison - {model_name}")
    print(f"{'='*70}")
    
    print(f"\n{'Metric':<30s} {'Vanilla':>12s} {'Formatted':>12s} {'Diff':>12s}")
    print("-" * 70)
    
    v_overall = vanilla_summary['overall']['percentage']
    f_overall = formatted_summary['overall']['percentage']
    improvement = f_overall - v_overall
    
    print(f"{'Overall':<30s} {v_overall:>11.1f}% {f_overall:>11.1f}% {improvement:>+11.1f}%")
    
    # 按类别对比
    print(f"\nBy Category:")
    all_categories = set(vanilla_summary['by_category'].keys()) | set(formatted_summary['by_category'].keys())
    for cat in sorted(all_categories):
        v_score = vanilla_summary['by_category'].get(cat, 0) * 100
        f_score = formatted_summary['by_category'].get(cat, 0) * 100
        diff = f_score - v_score
        print(f"  {cat:<28s} {v_score:>9.1f}% {f_score:>9.1f}% {diff:>+9.1f}%")
    
    # 按难度对比
    print(f"\nBy Difficulty:")
    all_difficulties = set(vanilla_summary['by_difficulty'].keys()) | set(formatted_summary['by_difficulty'].keys())
    for diff_level in sorted(all_difficulties):
        v_score = vanilla_summary['by_difficulty'].get(diff_level, 0) * 100
        f_score = formatted_summary['by_difficulty'].get(diff_level, 0) * 100
        diff = f_score - v_score
        print(f"  {diff_level:<28s} {v_score:>9.1f}% {f_score:>9.1f}% {diff:>+9.1f}%")
    
    # 保存对比报告
    comparison = {
        'model': model_name,
        'vanilla': vanilla_summary,
        'formatted': formatted_summary,
        'improvement': {
            'overall': improvement,
            'by_category': {
                cat: (formatted_summary['by_category'].get(cat, 0) - vanilla_summary['by_category'].get(cat, 0)) * 100
                for cat in all_categories
            },
            'by_difficulty': {
                diff: (formatted_summary['by_difficulty'].get(diff, 0) - vanilla_summary['by_difficulty'].get(diff, 0)) * 100
                for diff in all_difficulties
            }
        }
    }
    
    output_dir = Path("outputs") / model_name
    output_file = output_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Comparison saved: {output_file}")


def list_available_results():
    """列出所有可用的测试结果"""
    outputs_dir = Path("outputs")
    
    if not outputs_dir.exists():
        print("❌ No outputs directory found")
        return
    
    print(f"\n{'='*70}")
    print(f"📁 Available Test Results")
    print(f"{'='*70}\n")
    
    for model_dir in sorted(outputs_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        
        print(f"📦 {model_dir.name}:")
        
        for version_dir in sorted(model_dir.iterdir()):
            if not version_dir.is_dir():
                continue
            
            response_files = list(version_dir.glob("responses_*.json"))
            if response_files:
                latest = max(response_files, key=lambda p: p.stat().st_mtime)
                timestamp = datetime.fromtimestamp(latest.stat().st_mtime)
                print(f"   └─ {version_dir.name:12s} ({len(response_files)} files, latest: {timestamp.strftime('%Y-%m-%d %H:%M')})")
        
        print()


def show_help():
    """显示帮助信息"""
    help_text = """
Usage:
    python evaluate_results.py [options] <model_name> [version]
    python evaluate_results.py compare <model_name>
    python evaluate_results.py list

Options:
    <model_name>    Model name (e.g., gemini-3-pro-preview)
    [version]       Version name (vanilla or formatted), default: vanilla
    compare         Compare vanilla vs formatted versions
    list            List all available results

Examples:
    # Evaluate vanilla version
    python evaluate_results.py gemini-3-pro-preview vanilla
    
    # Evaluate formatted version
    python evaluate_results.py gemini-3-pro-preview formatted
    
    # Compare versions
    python evaluate_results.py compare gemini-3-pro-preview
    
    # List all results
    python evaluate_results.py list

Output:
    - Console: Summary statistics
    - File: Detailed JSON report in outputs/<model>/<version>/
    """
    print(help_text)


# ============================================
# 主程序
# ============================================

if __name__ == "__main__":
    # 检查qa.json
    if not Path("qa.json").exists():
        print("❌ Error: qa.json not found")
        print("   Please ensure qa.json is in the current directory")
        sys.exit(1)
    
    evaluator = SpreadsheetEvaluator("qa.json")
    
    if len(sys.argv) == 1:
        show_help()
        sys.exit(0)
    
    command = sys.argv[1]
    
    if command == "list":
        list_available_results()
    
    elif command == "compare":
        if len(sys.argv) < 3:
            print("❌ Error: Model name required")
            print("Usage: python evaluate_results.py compare <model_name>")
            sys.exit(1)
        
        model_name = sys.argv[2]
        compare_versions(model_name, evaluator)
    
    elif command == "help" or command == "-h" or command == "--help":
        show_help()
    
    else:
        # 评估单个版本
        model_name = sys.argv[1]
        version = sys.argv[2] if len(sys.argv) > 2 else "vanilla"
        
        evaluate_model_version(model_name, version, evaluator)
    
    print()