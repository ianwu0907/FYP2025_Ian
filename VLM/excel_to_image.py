"""
Excel转图片 - 绕过Visible属性问题
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
        print("✓ 已清理Excel进程")
    except:
        pass


def excel_to_image_workaround(excel_path: str, output_path: str):
    """
    绕过Visible属性问题的版本
    """
    excel_path = os.path.abspath(excel_path)
    output_path = os.path.abspath(output_path)
    
    print("=" * 60)
    print("🔧 Excel转图片 - Workaround版本")
    print("=" * 60)
    
    # 先清理Excel进程
    print("\n准备工作...")
    kill_excel_processes()
    
    pythoncom.CoInitialize()
    
    excel = None
    wb = None
    
    try:
        # 使用DispatchEx（创建新实例）
        print("\n1️⃣ 启动Excel（新实例）...")
        excel = win32com.client.DispatchEx("Excel.Application")
        
        # 不设置Visible属性，直接操作
        excel.DisplayAlerts = False
        
        time.sleep(0.5)
        
        # 打开文件
        print(f"2️⃣ 打开文件: {Path(excel_path).name}")
        wb = excel.Workbooks.Open(excel_path)
        ws = wb.Worksheets(1)
        
        print(f"   工作表: {ws.Name}")
        
        # 获取使用区域
        used_range = ws.UsedRange
        print(f"   区域: {used_range.Address}")
        
        # 方法A: PDF导出法（最可靠）
        print("\n3️⃣ 尝试PDF导出法...")
        pdf_path = output_path.replace('.png', '_temp.pdf')
        
        try:
            ws.ExportAsFixedFormat(
                Type=0,  # xlTypePDF
                Filename=pdf_path,
                Quality=0,
                IncludeDocProperties=True,
                IgnorePrintAreas=False,
                OpenAfterPublish=False
            )
            
            print(f"   ✓ PDF已生成")
            
            # 转换PDF为PNG
            try:
                from pdf2image import convert_from_path
                print("   转换PDF为PNG...")
                
                images = convert_from_path(pdf_path, dpi=200)
                
                if images:
                    images[0].save(output_path, 'PNG')
                    
                    # 删除临时PDF
                    try:
                        os.remove(pdf_path)
                    except:
                        pass
                    
                    size = os.path.getsize(output_path) / 1024
                    print(f"\n✅ 成功!")
                    print(f"   文件: {output_path}")
                    print(f"   大小: {size:.2f} KB")
                    return output_path
                    
            except ImportError:
                print(f"\n⚠️  需要安装: pip install pdf2image")
                print(f"   PDF文件已保存: {pdf_path}")
                print(f"\n你可以:")
                print(f"   1. 安装pdf2image: pip install pdf2image")
                print(f"   2. 或手动打开PDF并另存为PNG")
                return pdf_path
                
        except Exception as e:
            print(f"   ✗ PDF导出失败: {e}")
        
        # 方法B: Chart导出法（不设置Visible）
        print("\n4️⃣ 尝试Chart导出法...")
        try:
            # 复制
            used_range.CopyPicture(1, 2)
            time.sleep(2)
            
            # 创建图表
            chart = excel.Charts.Add()
            time.sleep(0.5)
            
            # 粘贴
            chart.Paste()
            time.sleep(2)
            
            # 导出
            chart.Export(output_path, "PNG")
            chart.Delete()
            
            if os.path.exists(output_path):
                size = os.path.getsize(output_path) / 1024
                
                if size > 5:
                    print(f"\n✅ 成功!")
                    print(f"   文件: {output_path}")
                    print(f"   大小: {size:.2f} KB")
                    return output_path
                else:
                    print(f"   ⚠️  文件很小 ({size:.2f} KB)，可能是空白")
            
        except Exception as e:
            print(f"   ✗ Chart导出失败: {e}")
        
        print("\n❌ 所有方法都失败了")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理
        print("\n清理资源...")
        try:
            if wb:
                wb.Close(SaveChanges=False)
        except:
            pass
        
        try:
            if excel:
                excel.Quit()
        except:
            pass
        
        pythoncom.CoUninitialize()
        
        # 再次清理进程
        time.sleep(1)
        kill_excel_processes()


def excel_to_pdf_only(excel_path: str, output_pdf: str = None):
    """
    只导出为PDF（不转PNG）
    """
    excel_path = os.path.abspath(excel_path)
    
    if output_pdf is None:
        output_pdf = Path(excel_path).stem + ".pdf"
    else:
        output_pdf = os.path.abspath(output_pdf)
    
    print("=" * 60)
    print("📄 Excel转PDF")
    print("=" * 60)
    
    kill_excel_processes()
    pythoncom.CoInitialize()
    
    try:
        print("\n1️⃣ 启动Excel...")
        excel = win32com.client.DispatchEx("Excel.Application")
        excel.DisplayAlerts = False
        
        print(f"2️⃣ 打开文件: {Path(excel_path).name}")
        wb = excel.Workbooks.Open(excel_path)
        ws = wb.Worksheets(1)
        
        print(f"   工作表: {ws.Name}")
        
        print("3️⃣ 导出为PDF...")
        ws.ExportAsFixedFormat(
            Type=0,
            Filename=output_pdf,
            Quality=0,
            IncludeDocProperties=True,
            IgnorePrintAreas=False,
            OpenAfterPublish=False
        )
        
        if os.path.exists(output_pdf):
            size = os.path.getsize(output_pdf) / 1024
            print(f"\n✅ 成功!")
            print(f"   文件: {output_pdf}")
            print(f"   大小: {size:.2f} KB")
            
            print(f"\n💡 提示:")
            print(f"   1. 安装 pdf2image 来自动转PNG:")
            print(f"      pip install pdf2image")
            print(f"   2. 然后运行:")
            print(f"      python {sys.argv[0]} {excel_path}")
        
        wb.Close(False)
        excel.Quit()
        
        return output_pdf
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pythoncom.CoUninitialize()
        kill_excel_processes()


def manual_screenshot_guide(excel_path: str):
    """
    打开Excel并提供手动截图指南
    """
    excel_path = os.path.abspath(excel_path)
    
    print("=" * 60)
    print("📖 手动截图指南")
    print("=" * 60)
    
    kill_excel_processes()
    pythoncom.CoInitialize()
    
    try:
        print("\n正在打开Excel...")
        excel = win32com.client.DispatchEx("Excel.Application")
        excel.DisplayAlerts = False
        
        # 打开文件
        wb = excel.Workbooks.Open(excel_path)
        ws = wb.Worksheets(1)
        
        # 尝试显示窗口
        try:
            excel.Visible = True
        except:
            print("⚠️  无法设置Excel为可见模式")
            print("   Excel可能在后台运行")
        
        # 选择数据区域
        ws.UsedRange.Select()
        
        print("\n" + "=" * 60)
        print("📸 请手动截图:")
        print("=" * 60)
        print("\n方法1 - Windows截图工具:")
        print("  1. 按 Win + Shift + S")
        print("  2. 框选Excel表格区域")
        print("  3. 图片会自动保存到剪贴板")
        print("  4. 打开Paint (Win键 → 画图)")
        print("  5. 粘贴 (Ctrl+V)")
        print("  6. 保存为PNG")
        
        print("\n方法2 - Snipping Tool:")
        print("  1. 打开截图工具 (Win键 → Snipping Tool)")
        print("  2. 新建 → 框选区域")
        print("  3. 文件 → 另存为 → PNG")
        
        print("\n按 Enter 键关闭Excel...")
        input()
        
        wb.Close(False)
        excel.Quit()
        
    except Exception as e:
        print(f"\n错误: {e}")
    finally:
        pythoncom.CoUninitialize()
        kill_excel_processes()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Excel转图片 - Workaround版本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自动转换（会尝试所有方法）
  python excel_workaround.py 2_Book1.xlsx
  
  # 只导出PDF
  python excel_workaround.py 2_Book1.xlsx --pdf-only
  
  # 打开Excel并提供手动截图指南
  python excel_workaround.py 2_Book1.xlsx --manual
        """
    )
    
    parser.add_argument('excel_file', help='Excel文件路径')
    parser.add_argument('-o', '--output', help='输出路径')
    parser.add_argument('--pdf-only', action='store_true',
                       help='只导出为PDF')
    parser.add_argument('--manual', action='store_true',
                       help='打开Excel并显示手动截图指南')
    
    args = parser.parse_args()
    
    if args.manual:
        manual_screenshot_guide(args.excel_file)
    elif args.pdf_only:
        output = args.output or f"{Path(args.excel_file).stem}.pdf"
        excel_to_pdf_only(args.excel_file, output)
    else:
        output = args.output or f"{Path(args.excel_file).stem}.png"
        excel_to_image_workaround(args.excel_file, output)