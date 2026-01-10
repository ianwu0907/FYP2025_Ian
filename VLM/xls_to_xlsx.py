"""
xls_to_xlsx_windows.py - Windows专用（完美保留格式）
"""
import os
import sys
from pathlib import Path

def convert_with_excel(xls_path: str, xlsx_path: str = None) -> str:
    """
    使用Excel COM对象转换（完美保留格式）
    仅Windows + Excel可用
    """
    if sys.platform != 'win32':
        raise OSError("此方法仅支持Windows系统")
    
    import win32com.client
    import pythoncom
    
    if xlsx_path is None:
        xlsx_path = xls_path.replace('.xls', '.xlsx')
    
    # 转为绝对路径
    xls_path = os.path.abspath(xls_path)
    xlsx_path = os.path.abspath(xlsx_path)
    
    print(f"📖 正在转换: {Path(xls_path).name}")
    
    pythoncom.CoInitialize()
    
    try:
        # 启动Excel
        excel = win32com.client.Dispatch("Excel.Application")
        excel.Visible = False
        excel.DisplayAlerts = False
        
        # 打开.xls文件
        wb = excel.Workbooks.Open(xls_path)
        
        # 另存为.xlsx
        # 51 = xlOpenXMLWorkbook (xlsx格式)
        wb.SaveAs(xlsx_path, FileFormat=51)
        
        # 关闭
        wb.Close()
        excel.Quit()
        
        print(f"✅ 已保存: {Path(xlsx_path).name}")
        print(f"   ✓ 完美保留所有格式")
        
        return xlsx_path
        
    finally:
        pythoncom.CoUninitialize()


def batch_convert_windows(input_dir: str, output_dir: str = None):
    """批量转换（Windows）"""
    if output_dir is None:
        output_dir = input_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    xls_files = [f for f in Path(input_dir).glob('*.xls') 
                 if f.suffix.lower() == '.xls']
    
    if not xls_files:
        print("❌ 未找到.xls文件")
        return
    
    print(f"🔍 找到 {len(xls_files)} 个文件")
    print("=" * 60)
    
    success = 0
    for i, xls_file in enumerate(xls_files, 1):
        print(f"\n[{i}/{len(xls_files)}]")
        try:
            output_path = os.path.join(output_dir, xls_file.stem + '.xlsx')
            convert_with_excel(str(xls_file), output_path)
            success += 1
        except Exception as e:
            print(f"❌ 失败: {e}")
    
    print("\n" + "=" * 60)
    print(f"✅ 成功: {success}/{len(xls_files)}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Windows Excel转换工具')
    parser.add_argument('input', nargs='?', help='.xls文件或文件夹')
    parser.add_argument('-o', '--output', help='输出路径')
    parser.add_argument('-d', '--directory', help='批量模式')
    
    args = parser.parse_args()
    
    if args.directory:
        batch_convert_windows(args.directory, args.output)
    elif args.input:
        if os.path.isdir(args.input):
            batch_convert_windows(args.input, args.output)
        elif os.path.isfile(args.input):
            convert_with_excel(args.input, args.output)
        else:
            print(f"❌ 路径不存在: {args.input}")
    else:
        parser.print_help()