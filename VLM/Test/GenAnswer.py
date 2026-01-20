import json
import base64
import time
from pathlib import Path
from typing import Dict, List
import google.generativeai as genai
from datetime import datetime
import os
from dotenv import load_dotenv

# ============================================
# 加载环境变量
# ============================================

# 从.env文件加载环境变量
load_dotenv()

# 获取环境变量
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file!")

# 设置代理（如果在.env中有配置）
HTTP_PROXY = os.getenv('HTTP_PROXY')
HTTPS_PROXY = os.getenv('HTTPS_PROXY')

if HTTP_PROXY:
    os.environ['HTTP_PROXY'] = HTTP_PROXY
if HTTPS_PROXY:
    os.environ['HTTPS_PROXY'] = HTTPS_PROXY

# 配置Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# ============================================
# 配置部分
# ============================================

QA_FILE = "qa.json"

# 模型配置
MODEL_NAME = os.getenv('MODEL_NAME', 'gemini-3-flash-preview')

# 版本配置 - 关键！
# 可选值: "vanilla" (无格式) 或 "formatted" (有格式)
VERSION = os.getenv('VERSION', 'vanilla')  # 从.env读取，默认vanilla

# 根据版本选择图片文件
IMAGE_FILES = {
    "vanilla": {
        "test_temp": "Test_vanilla.png",
        "test2_temp": "Test2_vanilla.png"
    },
    "formatted": {
        "test_temp": "Test_formatted.png",
        "test2_temp": "Test2_formatted.png"
    }
}

# 生成配置
GENERATION_CONFIG = {
    "temperature": float(os.getenv('TEMPERATURE', '0.7')),
    "top_p": float(os.getenv('TOP_P', '0.95')),
    "max_output_tokens": int(os.getenv('MAX_OUTPUT_TOKENS', '4096')),
}

# ============================================
# 文件管理
# ============================================

def get_model_directory(model_name: str, version: str) -> Path:
    """
    根据模型名称和版本创建并返回输出目录
    例如: gemini-3-pro-preview/vanilla/
    """
    # 创建outputs根目录
    base_dir = Path("outputs")
    base_dir.mkdir(exist_ok=True)
    
    # 创建模型专属目录
    model_dir = base_dir / model_name
    model_dir.mkdir(exist_ok=True)
    
    # 创建版本子目录
    version_dir = model_dir / version
    version_dir.mkdir(exist_ok=True)
    
    return version_dir

def get_progress_file(model_name: str, version: str) -> Path:
    """获取进度文件路径"""
    return get_model_directory(model_name, version) / "progress.json"

def get_responses_file(model_name: str, version: str) -> Path:
    """获取最终响应文件路径（带时间戳）"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return get_model_directory(model_name, version) / f"responses_{timestamp}.json"

def get_evaluation_file(model_name: str, version: str) -> Path:
    """获取评估报告文件路径（带时间戳）"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return get_model_directory(model_name, version) / f"evaluation_{timestamp}.json"

# ============================================
# 工具函数
# ============================================

def load_progress(model_name: str, version: str) -> Dict:
    """加载已完成的进度"""
    progress_file = get_progress_file(model_name, version)
    if progress_file.exists():
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_progress(responses: Dict, model_name: str, version: str):
    """保存进度"""
    progress_file = get_progress_file(model_name, version)
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)

def load_image(image_path: str) -> bytes:
    """加载图片"""
    with open(image_path, 'rb') as f:
        return f.read()

def load_qa_data(qa_file: str) -> Dict:
    """加载QA数据"""
    with open(qa_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def enhance_prompt_for_range(question: str, answer_type: str) -> str:
    """为range类型增强prompt"""
    if answer_type in ['range', 'cell_range']:
        return f"""{question}

**Important**: Please provide your answer in the format 'Cell1:Cell2' (e.g., A1:F10).
- Use the top-left cell and bottom-right cell notation
- Format: ColumnRow:ColumnRow (e.g., B2:E14)
- Be concise and direct"""
    return question

def call_gemini(image_path: str, question: str, answer_type: str, model_name: str) -> str:
    """
    调用Gemini API
    """
    try:
        # 加载图片
        image_data = load_image(image_path)
        
        # 增强prompt
        enhanced_question = enhance_prompt_for_range(question, answer_type)
        
        # 创建模型
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=GENERATION_CONFIG
        )
        
        # 调用API
        response = model.generate_content([
            enhanced_question,
            {"mime_type": "image/png", "data": image_data}
        ])
        
        return response.text
        
    except Exception as e:
        error_msg = str(e)
        print(f"       ❌ Error: {error_msg[:100]}")
        return f"ERROR: {error_msg}"

# ============================================
# 主程序
# ============================================

def main():
    print("=" * 70)
    print(f"Gemini API - Spreadsheet Understanding Evaluation")
    print(f"Model: {MODEL_NAME}")
    print(f"Version: {VERSION} ({'无格式' if VERSION == 'vanilla' else '有格式'})")
    print(f"Output Directory: ./outputs/{MODEL_NAME}/{VERSION}/")
    print("=" * 70)
    print()
    
    # 验证API key
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_API_KEY_HERE":
        print("❌ Error: Please set GEMINI_API_KEY in .env file")
        return
    
    print(f"✅ API Key loaded: {GEMINI_API_KEY[:10]}...{GEMINI_API_KEY[-4:]}")
    if HTTP_PROXY:
        print(f"✅ Proxy configured: {HTTP_PROXY}")
    print()
    
    # 验证版本配置
    if VERSION not in IMAGE_FILES:
        print(f"❌ Error: Invalid VERSION '{VERSION}'. Must be 'vanilla' or 'formatted'")
        return
    
    # 获取当前版本的图片配置
    current_images = IMAGE_FILES[VERSION]
    
    # 验证图片文件存在
    print("📷 Checking image files...")
    missing_files = []
    for file_id, image_path in current_images.items():
        if not Path(image_path).exists():
            missing_files.append(image_path)
            print(f"   ⚠️  Missing: {image_path}")
        else:
            print(f"   ✓ Found: {image_path}")
    
    if missing_files:
        print(f"\n❌ Error: {len(missing_files)} image file(s) not found!")
        return
    print()
    
    # 创建模型输出目录
    model_dir = get_model_directory(MODEL_NAME, VERSION)
    print(f"📁 Output directory: {model_dir}")
    print()
    
    # 加载QA数据
    print("📂 Loading QA data...")
    qa_data = load_qa_data(QA_FILE)
    total_questions = len(qa_data['qa_pairs'])
    print(f"   Loaded {total_questions} questions")
    
    # 加载进度
    print("📥 Loading progress...")
    vlm_responses = load_progress(MODEL_NAME, VERSION)
    completed = len(vlm_responses)
    print(f"   Already completed: {completed}/{total_questions}")
    print()
    
    if completed == total_questions:
        print("✅ All questions already completed!")
        print(f"📄 Results saved in: {model_dir}")
        return
    
    # 收集回答
    print(f"🤖 Calling {MODEL_NAME} API...")
    start_time = time.time()
    
    for i, qa in enumerate(qa_data['qa_pairs'], 1):
        qa_id = qa['id']
        
        # 跳过已完成的
        if qa_id in vlm_responses:
            print(f"   [{i}/{total_questions}] {qa_id} - ✅ Already done")
            continue
        
        file_id = qa['file_id']
        question = qa['question']
        answer_type = qa.get('answer_type', 'single_value')
        
        # 获取当前版本的图片
        image_path = current_images.get(file_id)
        if not image_path:
            print(f"   ⚠️  Warning: No image found for {file_id}")
            continue
        
        print(f"   [{i}/{total_questions}] {qa_id}...")
        print(f"       Image: {image_path}")
        print(f"       Q: {question[:50]}...")
        
        # 调用API
        response = call_gemini(image_path, question, answer_type, MODEL_NAME)
        vlm_responses[qa_id] = response
        
        # 显示回答（截断显示）
        display_answer = response[:80] + "..." if len(response) > 80 else response
        print(f"       A: {display_answer}")
        
        # 实时保存进度
        save_progress(vlm_responses, MODEL_NAME, VERSION)
        print(f"       💾 Progress saved ({len(vlm_responses)}/{total_questions})")
        print()
    
    # 计算耗时
    elapsed_time = time.time() - start_time
    
    # 保存最终结果
    final_file = get_responses_file(MODEL_NAME, VERSION)
    with open(final_file, 'w', encoding='utf-8') as f:
        json.dump(vlm_responses, f, indent=2, ensure_ascii=False)
    
    print("=" * 70)
    print("✅ All responses collected!")
    print(f"⏱️  Time elapsed: {elapsed_time:.1f} seconds")
    print(f"📊 Total questions: {total_questions}")
    print(f"📁 Results saved to: {model_dir}")
    print(f"   - Progress: {get_progress_file(MODEL_NAME, VERSION).name}")
    print(f"   - Responses: {final_file.name}")
    print("=" * 70)

if __name__ == "__main__":
    main()