"""
标准化处理 API 端点
处理表格标准化任务
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse
import uuid
import yaml
from pathlib import Path
import logging
import time
import pandas as pd
import json

from ...core.normalizer_wrapper import NormalizerWrapper
from ...models.request_models import NormalizeRequest
from ...models.response_models import TaskResponse, StatusResponse
from ...utils.dataframe_converter import DataFrameConverter
from .upload import session_manager  # 使用同一个 session_manager 实例

router = APIRouter(prefix="/normalize", tags=["normalize"])
logger = logging.getLogger(__name__)

# 任务状态存储（简化版，生产环境应使用 Redis）
tasks = {}

# 任务状态持久化目录
TASK_STORAGE_DIR = Path("storage/task_states")
TASK_STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def save_task_state(task_id: str, task_data: dict):
    """保存任务状态到文件"""
    try:
        task_file = TASK_STORAGE_DIR / f"{task_id}.json"
        with open(task_file, 'w', encoding='utf-8') as f:
            json.dump(task_data, f, ensure_ascii=False, indent=2)
        logger.debug(f"Task {task_id} state saved to {task_file}")
    except Exception as e:
        logger.error(f"Failed to save task state for {task_id}: {e}")


def load_task_state(task_id: str) -> dict:
    """从文件加载任务状态"""
    try:
        task_file = TASK_STORAGE_DIR / f"{task_id}.json"
        if task_file.exists():
            with open(task_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load task state for {task_id}: {e}")
    return None


@router.post("", response_model=TaskResponse)
async def start_normalization(
    normalize_request: NormalizeRequest,
    background_tasks: BackgroundTasks,
    request: Request
):
    """
    开始标准化处理

    Args:
        normalize_request: 标准化请求（包含 session_id 和配置覆盖）
        background_tasks: FastAPI 后台任务
        request: FastAPI Request 对象（用于获取主机信息）

    Returns:
        TaskResponse: 包含 task_id 和 WebSocket URL

    Raises:
        HTTPException: 会话不存在或没有上传文件
    """
    try:
        # 验证会话
        session = session_manager.get_session(normalize_request.session_id)
        if not session or not session.uploaded_file:
            raise HTTPException(
                status_code=404,
                detail="Session not found or no file uploaded"
            )

        logger.info(f"Starting normalization for session {normalize_request.session_id}")

        # 加载配置
        config_path = Path(__file__).parent.parent.parent.parent.parent / "spreadsheet-normalizer" / "config" / "config.yaml"

        if not config_path.exists():
            raise HTTPException(
                status_code=500,
                detail="Configuration file not found"
            )

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 应用配置覆盖
        if normalize_request.config_overrides:
            # 深度合并配置
            for key, value in normalize_request.config_overrides.items():
                if isinstance(value, dict) and key in config:
                    config[key].update(value)
                else:
                    config[key] = value

        logger.info(f"Loaded config with overrides: {list(normalize_request.config_overrides.keys())}")

        # 创建任务
        task_id = str(uuid.uuid4())
        task_data = {
            "status": "processing",
            "progress": 0,
            "session_id": normalize_request.session_id,
            "start_time": time.time()
        }
        tasks[task_id] = task_data
        save_task_state(task_id, task_data)  # 持久化

        # 在后台执行标准化
        background_tasks.add_task(
            run_normalization_task,
            task_id,
            str(session.uploaded_file),
            config
        )

        # 动态生成 WebSocket URL
        # 根据请求协议（HTTP/HTTPS）选择 ws/wss
        ws_scheme = "wss" if request.url.scheme == "https" else "ws"
        # 获取主机地址（包括端口）
        host = request.headers.get("host", "localhost:8000")
        websocket_url = f"{ws_scheme}://{host}/ws/progress/{task_id}"

        return TaskResponse(
            task_id=task_id,
            status="processing",
            message="Normalization started",
            websocket_url=websocket_url
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start normalization: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start normalization: {str(e)}"
        )


async def run_normalization_task(task_id: str, input_file: str, config: dict):
    """
    后台任务：执行标准化

    Args:
        task_id: 任务 ID
        input_file: 输入文件路径
        config: 配置字典
    """
    try:
        logger.info(f"Task {task_id}: Starting normalization of {input_file}")

        # 创建输出目录
        output_dir = Path("storage/outputs") / task_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # 设置输出路径
        config['logging']['output_dir'] = str(output_dir)

        # 定义进度回调
        def progress_callback(stage: str, progress: int, message: str):
            tasks[task_id]["progress"] = progress
            tasks[task_id]["current_stage"] = stage
            save_task_state(task_id, tasks[task_id])  # 持久化进度
            logger.info(f"Task {task_id}: [{stage}] {progress}% - {message}")

        # 创建包装器并执行
        wrapper = NormalizerWrapper(config, progress_callback=progress_callback)
        result = await wrapper.normalize_async(input_file)

        # 调试：打印返回结果
        logger.info(f"Task {task_id}: Normalizer result keys: {list(result.keys())}")

        # 读取标准化后的文件并生成预览
        normalized_preview = None
        output_path_obj = result.get("output_path")  # 可能是 Path 对象

        # 转换为 Path 对象（如果是字符串）
        if output_path_obj:
            if isinstance(output_path_obj, str):
                output_path_obj = Path(output_path_obj)
            elif not isinstance(output_path_obj, Path):
                output_path_obj = Path(str(output_path_obj))

        logger.info(f"Task {task_id}: Output path: '{output_path_obj}'")
        logger.info(f"Task {task_id}: Output path type: {type(output_path_obj)}")
        logger.info(f"Task {task_id}: Output path exists: {output_path_obj.exists() if output_path_obj else False}")

        if output_path_obj and output_path_obj.exists():
            try:
                logger.info(f"Task {task_id}: Reading normalized file for preview: {output_path_obj}")

                # 根据文件类型读取（使用 Path 的 suffix 属性）
                if output_path_obj.suffix.lower() == '.csv':
                    from .upload import read_csv_with_encoding_detection
                    df_normalized = read_csv_with_encoding_detection(output_path_obj)
                else:
                    from .upload import read_excel_file
                    df_normalized = read_excel_file(output_path_obj)

                # 生成预览
                normalized_preview = DataFrameConverter.to_json_preview(df_normalized, max_rows=100)
                logger.info(f"Task {task_id}: ✅ Successfully generated preview with shape {df_normalized.shape}")

            except Exception as e:
                logger.error(f"Task {task_id}: ❌ Failed to generate normalized preview: {e}", exc_info=True)
        else:
            logger.warning(f"Task {task_id}: ⚠️ Cannot generate preview - output_path empty or file not exists")

        # 更新任务状态
        elapsed_time = time.time() - tasks[task_id]["start_time"]
        task_data = {
            "status": "completed",
            "progress": 100,
            "result": {
                "output_path": str(output_path_obj) if output_path_obj else "",
                "num_tables": result.get("num_tables", 1),
                "detection_method": result.get("detection_method", "single"),
                "elapsed_seconds": elapsed_time,
                "normalized_preview": normalized_preview  # 添加标准化后的预览
            },
            "start_time": tasks[task_id]["start_time"],
            "end_time": time.time()
        }
        tasks[task_id] = task_data
        save_task_state(task_id, task_data)  # 持久化完成状态

        logger.info(f"Task {task_id}: ✅ Task completed with normalized_preview: {normalized_preview is not None}")

        logger.info(f"Task {task_id}: Completed successfully in {elapsed_time:.2f}s")

    except Exception as e:
        logger.error(f"Task {task_id}: Failed with error: {e}", exc_info=True)
        task_data = {
            "status": "failed",
            "progress": 0,
            "error": str(e),
            "start_time": tasks[task_id]["start_time"],
            "end_time": time.time()
        }
        tasks[task_id] = task_data
        save_task_state(task_id, task_data)  # 持久化失败状态


@router.get("/status/{task_id}", response_model=StatusResponse)
async def get_task_status(task_id: str):
    """
    获取任务状态

    Args:
        task_id: 任务 ID

    Returns:
        StatusResponse: 任务状态信息

    Raises:
        HTTPException: 任务不存在
    """
    # 先从内存查找
    task = tasks.get(task_id)

    # 如果内存中不存在，尝试从文件加载
    if task is None:
        task = load_task_state(task_id)
        if task:
            # 加载成功，缓存到内存
            tasks[task_id] = task
            logger.info(f"Task {task_id} loaded from persistent storage")

    # 如果仍然找不到，返回404
    if task is None:
        raise HTTPException(
            status_code=404,
            detail="Task not found"
        )

    # 计算已用时间
    elapsed_seconds = None
    if "start_time" in task:
        if task["status"] == "completed" or task["status"] == "failed":
            elapsed_seconds = task.get("end_time", time.time()) - task["start_time"]
        else:
            elapsed_seconds = time.time() - task["start_time"]

    return StatusResponse(
        task_id=task_id,
        status=task["status"],
        progress=task.get("progress", 0),
        current_stage=task.get("current_stage"),
        result=task.get("result"),
        error=task.get("error"),
        elapsed_seconds=elapsed_seconds
    )


@router.get("/download/{task_id}")
async def download_result(task_id: str):
    """
    下载标准化结果文件

    Args:
        task_id: 任务 ID

    Returns:
        FileResponse: CSV 文件

    Raises:
        HTTPException: 任务不存在或未完成
    """
    # 先从内存查找
    task = tasks.get(task_id)

    # 如果内存中不存在，尝试从文件加载
    if task is None:
        task = load_task_state(task_id)
        if task:
            tasks[task_id] = task
            logger.info(f"Task {task_id} loaded from persistent storage for download")

    # 如果仍然找不到，返回404
    if task is None:
        raise HTTPException(
            status_code=404,
            detail="Task not found"
        )

    # 检查任务是否完成
    if task["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Task is not completed yet (status: {task['status']})"
        )

    # 获取输出文件路径
    output_path = task.get("result", {}).get("output_path")
    if not output_path:
        raise HTTPException(
            status_code=404,
            detail="Output file path not found in task result"
        )

    output_path_obj = Path(output_path)

    # 检查文件是否存在
    if not output_path_obj.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Output file not found: {output_path}"
        )

    # 确定文件名
    filename = output_path_obj.name

    logger.info(f"Downloading file for task {task_id}: {output_path}")

    # 返回文件
    return FileResponse(
        path=str(output_path_obj),
        filename=filename,
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )
