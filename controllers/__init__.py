"""
控制器模块初始化文件
"""

from .auth_controller import auth_bp
from .jobfit_controller import jobfit_bp

__all__ = ['auth_bp', 'jobfit_bp']
