"""
自定义异常类 - 统一管理应用异常
特性：
1. 定义清晰的异常层次结构
2. 每个异常携带 HTTP 状态码
3. 便于全局异常处理和错误响应
"""

class JobFitException(Exception):
    """
    应用基础异常类
    """
    status_code = 500
    
    def __init__(self, message: str, status_code: int = None):
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
    
    def to_dict(self):
        """转换为字典格式，便于 JSON 响应"""
        return {
            'error': self.message,
            'status': self.status_code
        }

class ValidationException(JobFitException):
    """
    参数验证异常
    """
    status_code = 400

class AuthenticationException(JobFitException):
    """
    认证异常
    """
    status_code = 401

class AuthorizationException(JobFitException):
    """
    授权异常
    """
    status_code = 403

class NotFoundException(JobFitException):
    """
    资源未找到异常
    """
    status_code = 404

class FileUploadException(JobFitException):
    """
    文件上传异常
    """
    status_code = 400

class ModelException(JobFitException):
    """
    模型调用异常
    """
    status_code = 503

class DatabaseException(JobFitException):
    """
    数据库操作异常
    """
    status_code = 500
