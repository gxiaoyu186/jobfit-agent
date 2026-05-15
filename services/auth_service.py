"""
认证服务 - 处理用户注册、登录逻辑
特性：
1. 封装用户认证的核心业务逻辑
2. 使用配置中心的参数进行验证
3. 抛出统一的异常类型
4. 与数据存储解耦
"""

import csv
import os
from datetime import datetime

from config.settings import settings
from utils.crypto_utils import hash_password, verify_password, generate_token
from exceptions.jobfit_exceptions import (
    ValidationException,
    AuthenticationException,
    DatabaseException
)

class AuthService:
    """认证服务类"""
    
    def __init__(self):
        """初始化时确保用户文件存在"""
        self._ensure_users_file()
    
    def _ensure_users_file(self):
        """确保用户数据文件存在"""
        if not os.path.exists(settings.USERS_FILE):
            os.makedirs(settings.DATABASE_DIR, exist_ok=True)
            with open(settings.USERS_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['username', 'password_hash', 'created_at'])
    
    def register(self, username: str, password: str) -> dict:
        """
        用户注册
        
        Args:
            username: 用户名
            password: 密码
            
        Returns:
            注册结果字典
            
        Raises:
            ValidationException: 参数验证失败
            DatabaseException: 数据库操作失败
        """
        # 参数验证
        if not username or not password:
            raise ValidationException('用户名和密码不能为空')
        
        if len(username) < settings.USERNAME_MIN_LENGTH:
            raise ValidationException(
                f'用户名至少需要{settings.USERNAME_MIN_LENGTH}个字符'
            )
        
        if len(password) < settings.PASSWORD_MIN_LENGTH:
            raise ValidationException(
                f'密码至少需要{settings.PASSWORD_MIN_LENGTH}个字符'
            )
        
        # 检查用户是否已存在
        if self._user_exists(username):
            raise ValidationException('用户名已存在')
        
        # 创建用户
        try:
            password_hash = hash_password(password)
            created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            with open(settings.USERS_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([username, password_hash, created_at])
            
            return {'message': '注册成功', 'username': username}
        
        except Exception as e:
            raise DatabaseException(f'注册失败: {str(e)}')
    
    def login(self, username: str, password: str) -> dict:
        """
        用户登录
        
        Args:
            username: 用户名
            password: 密码
            
        Returns:
            登录结果字典（包含 token）
            
        Raises:
            ValidationException: 参数验证失败
            AuthenticationException: 认证失败
        """
        # 参数验证
        if not username or not password:
            raise ValidationException('用户名和密码不能为空')
        
        # 验证用户
        if not self._verify_user(username, password):
            raise AuthenticationException('用户名或密码错误')
        
        # 生成 token
        token = generate_token(16)
        
        return {
            'token': token,
            'username': username,
            'message': '登录成功'
        }
    
    def _user_exists(self, username: str) -> bool:
        """检查用户是否存在"""
        if not os.path.exists(settings.USERS_FILE):
            return False
        
        with open(settings.USERS_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['username'] == username:
                    return True
        return False
    
    def _verify_user(self, username: str, password: str) -> bool:
        """验证用户密码"""
        if not os.path.exists(settings.USERS_FILE):
            return False
        
        password_hash = hash_password(password)
        
        with open(settings.USERS_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['username'] == username and row['password_hash'] == password_hash:
                    return True
        return False
