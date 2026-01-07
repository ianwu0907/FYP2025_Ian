"""
会话管理器
管理用户上传文件的会话和任务状态
"""

from typing import Dict, Optional
import uuid
from datetime import datetime
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class Session:
    """会话类"""

    def __init__(self, session_id: str):
        """
        初始化会话

        Args:
            session_id: 会话 ID
        """
        self.session_id = session_id
        self.created_at = datetime.now()
        self.uploaded_file: Optional[Path] = None
        self.task_id: Optional[str] = None
        self.result: Optional[Dict] = None
        self.status: str = "created"  # created, uploaded, processing, completed, failed

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "uploaded_file": str(self.uploaded_file) if self.uploaded_file else None,
            "task_id": self.task_id,
            "status": self.status
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Session":
        """从字典创建会话对象"""
        session = cls(data["session_id"])
        session.created_at = datetime.fromisoformat(data["created_at"])
        session.uploaded_file = Path(data["uploaded_file"]) if data.get("uploaded_file") else None
        session.task_id = data.get("task_id")
        session.status = data.get("status", "created")
        return session


class SessionManager:
    """会话管理器"""

    def __init__(self, storage_dir: str = "storage/sessions"):
        """初始化会话管理器"""
        self.sessions: Dict[str, Session] = {}
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _save_session(self, session: Session):
        """保存会话到文件"""
        try:
            session_file = self.storage_dir / f"{session.session_id}.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
            logger.debug(f"Session {session.session_id} saved to {session_file}")
        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")

    def _load_session(self, session_id: str) -> Optional[Session]:
        """从文件加载会话"""
        try:
            session_file = self.storage_dir / f"{session_id}.json"
            if session_file.exists():
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return Session.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
        return None

    def create_session(self) -> str:
        """
        创建新会话

        Returns:
            会话 ID
        """
        session_id = str(uuid.uuid4())
        session = Session(session_id)
        self.sessions[session_id] = session
        self._save_session(session)  # 持久化
        return session_id

    def get_session(self, session_id: str) -> Optional[Session]:
        """
        获取会话

        Args:
            session_id: 会话 ID

        Returns:
            Session 对象或 None
        """
        # 先从内存查找
        session = self.sessions.get(session_id)

        # 如果内存中不存在，尝试从文件加载
        if session is None:
            session = self._load_session(session_id)
            if session:
                # 加载成功，缓存到内存
                self.sessions[session_id] = session
                logger.info(f"Session {session_id} loaded from persistent storage")

        return session

    def update_session(self, session_id: str, **kwargs):
        """
        更新会话属性

        Args:
            session_id: 会话 ID
            **kwargs: 要更新的属性
        """
        session = self.get_session(session_id)
        if session:
            for key, value in kwargs.items():
                if hasattr(session, key):
                    setattr(session, key, value)
            self._save_session(session)  # 持久化更新

    def delete_session(self, session_id: str) -> bool:
        """
        删除会话

        Args:
            session_id: 会话 ID

        Returns:
            是否删除成功
        """
        # 从内存删除
        if session_id in self.sessions:
            del self.sessions[session_id]

        # 从文件删除
        try:
            session_file = self.storage_dir / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
                logger.info(f"Session {session_id} file deleted")
                return True
        except Exception as e:
            logger.error(f"Failed to delete session file {session_id}: {e}")

        return False

    def cleanup_expired_sessions(self, timeout_seconds: int = 3600):
        """
        清理过期会话

        Args:
            timeout_seconds: 超时时间（秒）
        """
        now = datetime.now()
        expired_ids = [
            sid for sid, session in self.sessions.items()
            if (now - session.created_at).total_seconds() > timeout_seconds
        ]

        for sid in expired_ids:
            self.delete_session(sid)

        return len(expired_ids)
