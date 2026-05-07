"""
加密工具函数 - 提供密码加密和 token 生成功能
特性：
1. 使用 SHA-256 进行密码哈希
2. 提供安全的随机 token 生成
3. 统一管理加密相关逻辑
"""

import hashlib
import secrets

def hash_password(password: str) -> str:
    """
    使用 SHA-256 哈希密码
    
    Args:
        password: 原始密码
        
    Returns:
        哈希后的密码字符串
    """
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    验证密码是否匹配
    
    Args:
        plain_password: 用户输入的密码
        hashed_password: 存储的哈希密码
        
    Returns:
        是否匹配
    """
    return hash_password(plain_password) == hashed_password

def generate_token(length: int = 32) -> str:
    """
    生成安全的随机 token
    
    Args:
        length: token 长度（字节数）
        
    Returns:
        十六进制格式的 token 字符串
    """
    return secrets.token_hex(length)

def generate_session_id() -> str:
    """
    生成会话 ID
    
    Returns:
        会话 ID 字符串
    """
    return secrets.token_urlsafe(16)
