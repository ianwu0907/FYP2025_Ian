"""
文件上传 API 端点
处理 Excel/CSV 文件上传并生成预览
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import pandas as pd
from datetime import datetime
import logging
import chardet

from ...core.session_manager import SessionManager
from ...models.response_models import UploadResponse
from ...utils.dataframe_converter import DataFrameConverter

router = APIRouter(prefix="/upload", tags=["upload"])
logger = logging.getLogger(__name__)

# 全局会话管理器实例
session_manager = SessionManager()


def read_csv_with_encoding_detection(file_path: Path) -> pd.DataFrame:
    """
    自动检测 CSV 文件编码并读取

    Args:
        file_path: CSV 文件路径

    Returns:
        pd.DataFrame: 读取的数据框
    """
    # 读取文件的前几个字节来检测编码
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)  # 读取前 10KB

    # 使用 chardet 检测编码
    detected = chardet.detect(raw_data)
    encoding = detected['encoding']
    confidence = detected['confidence']

    logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2%})")

    # 尝试使用检测到的编码
    encodings_to_try = [encoding, 'utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']

    for enc in encodings_to_try:
        if enc is None:
            continue
        try:
            logger.info(f"Trying to read CSV with encoding: {enc}")
            df = pd.read_csv(file_path, encoding=enc)
            logger.info(f"Successfully read CSV with encoding: {enc}")
            return df
        except (UnicodeDecodeError, LookupError) as e:
            logger.warning(f"Failed to read with encoding {enc}: {e}")
            continue

    # 如果所有编码都失败，尝试使用 errors='ignore'
    logger.warning("All encoding attempts failed, using utf-8 with errors='ignore'")
    return pd.read_csv(file_path, encoding='utf-8', errors='ignore')


def read_excel_file(file_path: Path) -> pd.DataFrame:
    """
    读取 Excel 文件（支持 .xlsx 和 .xls）

    Args:
        file_path: Excel 文件路径

    Returns:
        pd.DataFrame: 读取的数据框
    """
    # 尝试不同的引擎
    engines = ['openpyxl', 'xlrd']  # openpyxl for .xlsx, xlrd for .xls

    for engine in engines:
        try:
            logger.info(f"Trying to read Excel with engine: {engine}")
            df = pd.read_excel(file_path, engine=engine)
            logger.info(f"Successfully read Excel with engine: {engine}")
            return df
        except Exception as e:
            logger.warning(f"Failed to read with engine {engine}: {e}")
            continue

    # 如果都失败了，抛出异常
    raise ValueError("Failed to read Excel file with all available engines")


@router.post("", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    上传 Excel/CSV 文件

    Args:
        file: 上传的文件（.xlsx 或 .csv）

    Returns:
        UploadResponse: 包含 session_id 和文件预览

    Raises:
        HTTPException: 文件类型不支持或上传失败
    """
    try:
        # 验证文件类型
        if not file.filename.endswith(('.xlsx', '.csv', '.xls')):
            raise HTTPException(
                status_code=400,
                detail="Only .xlsx, .xls, and .csv files are supported"
            )

        # 创建会话
        session_id = session_manager.create_session()
        session = session_manager.get_session(session_id)

        logger.info(f"Created session {session_id} for file {file.filename}")

        # 保存文件
        upload_dir = Path("storage/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / f"{session_id}_{file.filename}"

        # 读取并保存文件内容
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"Saved file to {file_path}")

        # 更新会话
        session_manager.update_session(session_id, uploaded_file=file_path, status="uploaded")

        # 生成预览
        try:
            if file.filename.endswith('.csv'):
                # 自动检测 CSV 文件编码
                df = read_csv_with_encoding_detection(file_path)
            else:
                # 读取 Excel 文件（支持 .xlsx 和 .xls）
                df = read_excel_file(file_path)

            preview = DataFrameConverter.to_json_preview(df, max_rows=10)
            logger.info(f"Generated preview for {file.filename}: shape={df.shape}")

        except Exception as e:
            logger.error(f"Failed to generate preview: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read file: {str(e)}"
            )

        return UploadResponse(
            session_id=session_id,
            filename=file.filename,
            file_type=file_path.suffix[1:],  # 移除 '.'
            file_size=len(content),
            preview=preview,
            upload_time=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )


@router.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """
    获取会话信息

    Args:
        session_id: 会话 ID

    Returns:
        会话信息字典

    Raises:
        HTTPException: 会话不存在
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )

    return session.to_dict()
