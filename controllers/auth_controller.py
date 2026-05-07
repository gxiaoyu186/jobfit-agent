"""
认证控制器 - 处理认证相关的 HTTP 请求
特性：
1. 仅负责处理 HTTP 请求和响应
2. 调用服务层处理业务逻辑
3. 统一异常处理和响应格式
"""

from flask import Blueprint, request, jsonify
from services.auth_service import AuthService
from exceptions.jobfit_exceptions import JobFitException

# 创建蓝图
auth_bp = Blueprint('auth', __name__, url_prefix='/api/v1/auth')

# 初始化服务
auth_service = AuthService()

@auth_bp.route('/register', methods=['POST'])
def register():
    """
    用户注册接口
    
    请求体：
    {
        "username": "用户名",
        "password": "密码"
    }
    
    响应：
    {
        "message": "注册成功",
        "username": "用户名"
    }
    """
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        result = auth_service.register(username, password)
        return jsonify(result), 200
    
    except JobFitException as e:
        return jsonify({'error': e.message}), e.status_code
    except Exception as e:
        return jsonify({'error': '服务器内部错误'}), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    """
    用户登录接口
    
    请求体：
    {
        "username": "用户名",
        "password": "密码"
    }
    
    响应：
    {
        "token": "xxx",
        "username": "用户名",
        "message": "登录成功"
    }
    """
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        result = auth_service.login(username, password)
        return jsonify(result), 200
    
    except JobFitException as e:
        return jsonify({'error': e.message}), e.status_code
    except Exception as e:
        return jsonify({'error': '服务器内部错误'}), 500
