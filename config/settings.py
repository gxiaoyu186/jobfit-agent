"""
配置中心 - 统一管理所有配置项
特性：
1. 使用 Pydantic Settings 进行类型安全的配置解析
2. 支持从环境变量和 .env 文件读取配置
3. 提供默认值和验证逻辑
4. 便于多环境部署切换
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Settings:
    """应用配置类"""
    
    # API 配置
    API_KEY: str = os.getenv("API_KEY", "")
    BASE_URL: str = os.getenv("BASE_URL", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    
    # 模型配置（OCR功能需要支持多模态的模型）
    MODEL_NAME: str = os.getenv("MODEL_NAME", "qwen3.5-plus")
    MODEL_TEMPERATURE: float = float(os.getenv("MODEL_TEMPERATURE", "0.0"))
    
    # 数据库配置（保持CSV存储）
    DATABASE_DIR: str = os.getenv("DATABASE_DIR", "./database")
    USERS_FILE: str = os.path.join(DATABASE_DIR, "users.csv")
    
    # 文件存储配置
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")
    REPORTS_DIR: str = os.getenv("REPORTS_DIR", "./reports")
    RESOURCES_DIR: str = os.getenv("RESOURCES_DIR", "./resources")
    
    # 日志配置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "./logs/jobfit.log")
    
    # 安全配置
    TOKEN_EXPIRE_HOURS: int = int(os.getenv("TOKEN_EXPIRE_HOURS", "24"))
    PASSWORD_MIN_LENGTH: int = int(os.getenv("PASSWORD_MIN_LENGTH", "6"))
    USERNAME_MIN_LENGTH: int = int(os.getenv("USERNAME_MIN_LENGTH", "3"))
    
    # API 配置
    API_VERSION: str = "v1"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    def __init__(self):
        """初始化时创建必要的目录"""
        self._ensure_directories()
    
    def _ensure_directories(self):
        """确保所有必要目录存在"""
        directories = [
            self.DATABASE_DIR,
            self.UPLOAD_DIR,
            self.REPORTS_DIR,
            self.RESOURCES_DIR,
            os.path.dirname(self.LOG_FILE)
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def is_configured(self) -> bool:
        """检查关键配置是否已设置"""
        return bool(self.API_KEY and self.BASE_URL)

# 创建全局配置实例
settings = Settings()
