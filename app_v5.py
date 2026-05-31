"""
JobFit Agent 优化版入口文件
特性：
1. 使用蓝图架构，代码结构清晰
2. 统一异常处理
3. 配置中心化
4. 支持 API 版本控制
5. 保留原有功能，同时提升可维护性
"""

from flask import Flask, jsonify, send_from_directory
from config.settings import settings

from controllers.auth_controller import auth_bp
from controllers.jobfit_controller import jobfit_bp
from exceptions.jobfit_exceptions import JobFitException

def create_app():
    """创建 Flask 应用实例"""
    app = Flask(__name__, static_folder='frontend')
    
    # CORS 配置 - 允许所有来源的跨域请求
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response
    
    # 注册蓝图
    app.register_blueprint(auth_bp)
    app.register_blueprint(jobfit_bp)
    
    # 全局异常处理
    @app.errorhandler(JobFitException)
    def handle_jobfit_exception(e):
        return jsonify({'error': e.message}), e.status_code
    
    @app.errorhandler(404)
    def handle_not_found(e):
        return jsonify({'error': '页面未找到'}), 404
    
    @app.errorhandler(500)
    def handle_internal_error(e):
        return jsonify({'error': '服务器内部错误'}), 500
    
    # 静态页面路由
    @app.route('/')
    def index():
        return send_from_directory('frontend', 'index.html')
    
    @app.route('/index.html')
    def index_page():
        return send_from_directory('frontend', 'index.html')
    
    @app.route('/register.html')
    def register_page():
        return send_from_directory('frontend', 'register.html')
    
    @app.route('/dashboard')
    def dashboard():
        return send_from_directory('frontend', 'dashboard.html')
    
    @app.route('/report.html')
    def report_page():
        return send_from_directory('frontend', 'report.html')
    
    return app

if __name__ == '__main__':
    app = create_app()
    
    print('=' * 50)
    print('JobFit Agent 后端服务 v5.0 已启动')
    print(f'配置检查: {"[OK] 已配置" if settings.is_configured() else "[ERROR] 缺少关键配置"}')
    print('访问地址: http://localhost:5000')
    print('=' * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=False)