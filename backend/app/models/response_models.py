"""
Pydantic 响应模型
定义 API 响应的数据结构
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class UploadResponse(BaseModel):
    """文件上传响应模型"""

    session_id: str = Field(..., description="Session ID for tracking")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File type (xlsx or csv)")
    file_size: int = Field(..., description="File size in bytes")
    preview: Dict[str, Any] = Field(..., description="Preview of the uploaded file")
    upload_time: str = Field(..., description="Upload timestamp (ISO format)")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "filename": "testdata.xlsx",
                "file_type": "xlsx",
                "file_size": 55194,
                "preview": {
                    "shape": [100, 15],
                    "columns": ["Column1", "Column2"],
                    "data": [{"Column1": "Value1"}]
                },
                "upload_time": "2026-01-05T10:30:00Z"
            }
        }


class TaskResponse(BaseModel):
    """任务响应模型"""

    task_id: str = Field(..., description="Task ID for tracking")
    status: str = Field(..., description="Task status: processing, completed, failed")
    message: str = Field(..., description="Status message")
    websocket_url: Optional[str] = Field(default=None, description="WebSocket URL for progress updates")

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "task-uuid-123",
                "status": "processing",
                "message": "Normalization started",
                "websocket_url": "ws://localhost:8000/ws/progress/task-uuid-123"
            }
        }


class StatusResponse(BaseModel):
    """任务状态响应模型"""

    task_id: str = Field(..., description="Task ID")
    status: str = Field(..., description="Task status")
    progress: int = Field(..., description="Progress percentage (0-100)")
    current_stage: Optional[str] = Field(default=None, description="Current processing stage")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Result data (if completed)")
    error: Optional[str] = Field(default=None, description="Error message (if failed)")
    elapsed_seconds: Optional[float] = Field(default=None, description="Elapsed time in seconds")

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "task-uuid-123",
                "status": "completed",
                "progress": 100,
                "current_stage": "Transformation",
                "result": {
                    "num_tables": 2,
                    "output_files": [
                        {"table_name": "Sheet1", "file_id": "file-uuid-1"}
                    ]
                },
                "elapsed_seconds": 83.5
            }
        }


class ErrorResponse(BaseModel):
    """错误响应模型"""

    error: str = Field(..., description="Error code or type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "SESSION_NOT_FOUND",
                "message": "Session does not exist or has expired",
                "details": {"session_id": "invalid-uuid"}
            }
        }
