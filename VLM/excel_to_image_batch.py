"""
Excel批量转图片 - 只处理.xlsx文件
输出到picture文件夹
"""

import os
import sys
import win32com.client
import pythoncom
from pathlib import Path
import time
import subprocess

def kill_excel_processes():
    """强制关闭所有Excel进程"""
    try:
        subprocess.run(['taskkill', '/F', '/IM', 'excel.exe'], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL)
        time.sleep(1)
    except:
        pass


def excel_to_image_single(excel_path: str, output_path: str, method='auto'):
    """
    转换单个Excel文件为图片
    
    Args:
        excel_path: Excel文件路径
        output_path: 输出图片路径
        method: 'pdf' 或 'chart' 或 'auto'（自动选择）
    
    Returns:
        成功返回输出路径，失败返回None
    """
    excel_path = os.path.abspath(excel_path)
    output_path = os.path.abspath(output_path)
    
    pythoncom.CoInitialize()
    
    excel = None
    wb = None
    
    try:
        # 使用DispatchEx（创建新实例）
        excel = win32com.client.DispatchEx("Excel.Application")
        excel.DisplayAlerts = False
        
        # 打开文件
        wb = excel.Workbooks.Open(excel_path)
        ws = wb.Worksheets(1)
        
        # 获取使用区域
        used_range = ws.UsedRange
        
        success = False
        
        # 方法1: PDF导出法（最可靠）
        if method in ['auto', 'pdf']:
            try:
                pdf_path = output_path.replace('.png', '_temp.pdf')
                
                ws.ExportAsFixedFormat(
                    Type=0,  # xlTypePDF
                    Filename=pdf_path,
                    Quality=0,
                    IncludeDocProperties=True,
                    IgnorePrintAreas=False,
                    OpenAfterPublish=False
                )
                
                # 转换PDF为PNG
                try:
                    from pdf2image import convert_from_path
                    images = convert_from_path(pdf_path, dpi=200)
                    
                    if images:
                        images[0].save(output_path, 'PNG')
                        
                        # 删除临时PDF
                        try:
                            os.remove(pdf_path)
                        except:
                            pass
                        
                        success = True
                        
                except ImportError:
                    # 如果没有pdf2image，保留PDF
                    if os.path.exists(pdf_path):
                        # 重命名为最终输出
                        final_pdf = output_path.replace('.png', '.pdf')
                        os.rename(pdf_path, final_pdf)
                        output_path = final_pdf
                        success = True
                    
            except Exception as e:
                pass
        
        # 方法2: Chart导出法（备选）
        if not success and method in ['auto', 'chart']:
            try:
                used_range.CopyPicture(1, 2)
                time.sleep(1)
                
                chart = excel.Charts.Add()
                chart.Paste()
                time.sleep(1)
                
                chart.Export(output_path, "PNG")
                chart.Delete()
                
                if os.path.exists(output_path):
                    size = os.path.getsize(output_path) / 1024
                    if size > 5:
                        success = True
                
            except Exception as e:
                pass
        
        # 清理
        wb.Close(SaveChanges=False)
        excel.Quit()
        
        pythoncom.CoUninitialize()
        
        if success:
            return output_path
        else:
            return None
            
    except Exception as e:
        if wb:
            try:
                wb.Close(SaveChanges=False)
            except:
                pass
        if excel:
            try:
                excel.Quit()
            except:
                pass
        
        pythoncom.CoUninitialize()
        return None


def batch_convert_excel_to_image(input_dir: str, 
                                 output_dir: str = 'picture',
                                 method: str = 'auto',
                                 skip_existing: bool = True,
                                 extensions: list = None):
    """
    批量转换Excel文件为图片
    
    Args:
        input_dir: 输入文件夹路径
        output_dir: 输出文件夹路径（默认'picture'）
        method: 转换方法 ('auto', 'pdf', 'chart')
        skip_existing: 是否跳过已存在的文件
        extensions: 要处理的文件扩展名列表（默认只处理.xlsx）
    """
    # 清理Excel进程
    print("准备工作：清理Excel进程...")
    kill_excel_processes()
    
    # 确保输入目录存在
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ 输入文件夹不存在: {input_dir}")
        return
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ 输出文件夹: {output_path.absolute()}")
    
    # 默认只处理.xlsx文件
    if extensions is None:
        extensions = ['*.xlsx']
    
    # 查找指定类型的Excel文件
    excel_files = []
    for ext in extensions:
        excel_files.extend(input_path.glob(ext))
    
    if not excel_files:
        print(f"❌ 在 {input_dir} 中未找到.xlsx文件")
        return
    
    print(f"\n🔍 找到 {len(excel_files)} 个.xlsx文件")
    
    # 统计xls文件数量（信息提示）
    xls_files = list(input_path.glob('*.xls'))
    # 排除xlsx文件
    xls_only = [f for f in xls_files if f.suffix.lower() == '.xls']
    if xls_only:
        print(f"💡 跳过 {len(xls_only)} 个.xls文件（只处理.xlsx）")
    
    print("=" * 60)
    
    success_count = 0
    skip_count = 0
    fail_count = 0
    failed_files = []
    
    for i, excel_file in enumerate(excel_files, 1):
        print(f"\n[{i}/{len(excel_files)}] {excel_file.name}")
        
        # 生成输出文件名
        output_file = output_path / f"{excel_file.stem}.png"
        
        # 检查是否已存在
        if skip_existing and output_file.exists():
            print(f"   ⏭️  已存在，跳过")
            skip_count += 1
            continue
        
        # 转换
        print(f"   🔄 转换中...")
        
        try:
            result = excel_to_image_single(str(excel_file), str(output_file), method)
            
            if result:
                size = os.path.getsize(result) / 1024
                print(f"   ✅ 成功! ({size:.2f} KB)")
                success_count += 1
            else:
                print(f"   ❌ 失败")
                fail_count += 1
                failed_files.append(excel_file.name)
                
        except Exception as e:
            print(f"   ❌ 错误: {e}")
            fail_count += 1
            failed_files.append(excel_file.name)
        
        # 每5个文件清理一次Excel进程（防止内存泄漏）
        if i % 5 == 0:
            kill_excel_processes()
    
    # 最终清理
    print("\n清理Excel进程...")
    kill_excel_processes()
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 转换完成!")
    print("=" * 60)
    print(f"✅ 成功: {success_count}")
    print(f"⏭️  跳过: {skip_count}")
    print(f"❌ 失败: {fail_count}")
    print(f"📁 输出文件夹: {output_path.absolute()}")
    
    if failed_files:
        print(f"\n失败的文件:")
        for f in failed_files:
            print(f"  - {f}")
    
    return success_count, skip_count, fail_count


def batch_convert_with_progress(input_dir: str, 
                                output_dir: str = 'picture',
                                method: str = 'auto',
                                extensions: list = None):
    """
    带进度条的批量转换（需要tqdm）
    """
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False
        print("💡 提示: 安装tqdm可显示进度条")
        print("   pip install tqdm\n")
    
    # 清理Excel进程
    kill_excel_processes()
    
    # 确保输入目录存在
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ 输入文件夹不存在: {input_dir}")
        return
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 默认只处理.xlsx文件
    if extensions is None:
        extensions = ['*.xlsx']
    
    # 查找Excel文件
    excel_files = []
    for ext in extensions:
        excel_files.extend(input_path.glob(ext))
    
    if not excel_files:
        print(f"❌ 未找到.xlsx文件")
        return
    
    print(f"🔍 找到 {len(excel_files)} 个.xlsx文件")
    
    # 统计xls文件
    xls_only = [f for f in input_path.glob('*.xls') if f.suffix.lower() == '.xls']
    if xls_only:
        print(f"💡 跳过 {len(xls_only)} 个.xls文件")
    
    print(f"📁 输出到: {output_path.absolute()}")
    print("=" * 60)
    
    success_count = 0
    fail_count = 0
    failed_files = []
    
    # 使用进度条（如果有tqdm）
    iterator = tqdm(excel_files, desc="转换中") if has_tqdm else excel_files
    
    for i, excel_file in enumerate(iterator, 1):
        if not has_tqdm:
            print(f"[{i}/{len(excel_files)}] {excel_file.name}")
        
        output_file = output_path / f"{excel_file.stem}.png"
        
        try:
            result = excel_to_image_single(str(excel_file), str(output_file), method)
            
            if result:
                success_count += 1
                if not has_tqdm:
                    print(f"  ✅ 成功")
            else:
                fail_count += 1
                failed_files.append(excel_file.name)
                if not has_tqdm:
                    print(f"  ❌ 失败")
                    
        except Exception as e:
            fail_count += 1
            failed_files.append(excel_file.name)
            if not has_tqdm:
                print(f"  ❌ 错误: {e}")
        
        # 定期清理
        if i % 5 == 0:
            kill_excel_processes()
    
    kill_excel_processes()
    
    # 总结
    print("\n" + "=" * 60)
    print("✅ 转换完成!")
    print(f"   成功: {success_count}/{len(excel_files)}")
    if fail_count > 0:
        print(f"   失败: {fail_count}")
        print("\n失败的文件:")
        for f in failed_files:
            print(f"  - {f}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Excel批量转图片工具（只处理.xlsx文件）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础用法：转换dataset文件夹中的.xlsx，输出到picture
  python excel_to_image.py dataset
  
  # 指定输出文件夹
  python excel_to_image.py dataset -o images
  
  # 使用进度条模式
  python excel_to_image.py dataset --progress
  
  # 指定转换方法
  python excel_to_image.py dataset -m pdf
  
  # 转换单个文件
  python excel_to_image.py file.xlsx
  
  # 也处理.xls和.xlsm文件（添加额外格式）
  python excel_to_image.py dataset --include-xls --include-xlsm
        """
    )
    
    parser.add_argument('input', help='输入文件或文件夹')
    parser.add_argument('-o', '--output', default='picture',
                       help='输出文件夹（默认: picture）')
    parser.add_argument('-m', '--method', 
                       choices=['auto', 'pdf', 'chart'],
                       default='auto',
                       help='转换方法（默认: auto）')
    parser.add_argument('--progress', action='store_true',
                       help='使用进度条模式（需要tqdm）')
    parser.add_argument('--no-skip', action='store_true',
                       help='不跳过已存在的文件')
    parser.add_argument('--include-xls', action='store_true',
                       help='也处理.xls文件')
    parser.add_argument('--include-xlsm', action='store_true',
                       help='也处理.xlsm文件')
    
    args = parser.parse_args()
    
    # 确定要处理的文件扩展名
    extensions = ['*.xlsx']
    if args.include_xls:
        extensions.append('*.xls')
    if args.include_xlsm:
        extensions.append('*.xlsm')
    
    input_path = Path(args.input)
    
    # 判断是文件还是文件夹
    if input_path.is_file():
        # 单文件模式
        print(f"转换单个文件: {input_path.name}")
        
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{input_path.stem}.png"
        
        kill_excel_processes()
        result = excel_to_image_single(str(input_path), str(output_file), args.method)
        
        if result:
            size = os.path.getsize(result) / 1024
            print(f"✅ 成功! 文件: {result} ({size:.2f} KB)")
        else:
            print(f"❌ 转换失败")
        
        kill_excel_processes()
        
    elif input_path.is_dir():
        # 批量模式
        if args.progress:
            batch_convert_with_progress(args.input, args.output, args.method, extensions)
        else:
            batch_convert_excel_to_image(
                args.input, 
                args.output, 
                args.method,
                skip_existing=not args.no_skip,
                extensions=extensions
            )
    else:
        print(f"❌ 路径不存在: {args.input}")