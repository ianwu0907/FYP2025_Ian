"""
FastAPI 主应用入口
Spreadsheet Normalizer API
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import logging

from .api.routes import upload, normalize

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="Spreadsheet Normalizer API",
    description="API for normalizing messy spreadsheets using LLM-powered analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 配置
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info(f"CORS enabled for origins: {cors_origins}")

# 注册路由
app.include_router(upload.router, prefix="/api/v1")
app.include_router(normalize.router, prefix="/api/v1")

logger.info("API routes registered")


@app.get("/")
async def root():
    """根端点"""
    return {
        "message": "Spreadsheet Normalizer API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }


@app.get("/health")
async def health():
    """健康检查端点"""
    return {
        "status": "healthy",
        "service": "Spreadsheet Normalizer API"
    }


@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("=" * 80)
    logger.info("Spreadsheet Normalizer API Starting...")
    logger.info("=" * 80)
    logger.info(f"API Documentation: http://localhost:8000/docs")
    logger.info(f"Health Check: http://localhost:8000/health")
    logger.info("=" * 80)


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("Spreadsheet Normalizer API Shutting down...")
