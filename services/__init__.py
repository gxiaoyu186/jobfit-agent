"""
服务模块初始化文件
"""

from .auth_service import AuthService
from .jobfit_service import JobFitService, AnalysisResult

__all__ = ['AuthService', 'JobFitService', 'AnalysisResult']
