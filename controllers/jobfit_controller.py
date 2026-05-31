"""
JobFit 控制器 - 处理匹配分析相关的 HTTP 请求
特性：
1. 仅负责处理 HTTP 请求和响应
2. 支持流式响应
3. 统一异常处理和响应格式
"""

import json
import sys
from flask import Blueprint, request, jsonify, Response
from services.jobfit_service import JobFitService
from config.settings import settings
from exceptions.jobfit_exceptions import JobFitException, FileUploadException, ValidationException
from datetime import datetime
import os

# 创建蓝图
jobfit_bp = Blueprint('jobfit', __name__, url_prefix='/api/v1')

# 初始化服务
jobfit_service = JobFitService()

@jobfit_bp.route('/test', methods=['GET'])
def test():
    """测试接口"""
    return jsonify({'message': '测试成功'}), 200

@jobfit_bp.route('/agent/chat', methods=['POST'])
def agent_chat():
    """
    Agent 对话接口（流式响应）
    
    请求体：
    {
        "message": "用户消息",
        "resume_path": "简历文件路径（可选）",
        "jd_path": "JD文件路径（可选）",
        "username": "用户名（可选，用于生成报告）"
    }
    
    用户可以上传简历和JD图片，同时可以在消息中输入额外的文字说明，
    系统会综合图片识别结果和用户消息进行分析。
    
    响应：Server-Sent Events (SSE)
    """
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        resume_path = data.get('resume_path', '')
        jd_path = data.get('jd_path', '')
        username = data.get('username', 'anonymous')
        
        def generate():
            try:
                yield ":ok\n\n"
                for chunk in jobfit_service.analyze_match(
                    resume_path=resume_path,
                    jd_path=jd_path,
                    message=message,
                    username=username
                ):
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                
            except JobFitException as e:
                yield f"data: {json.dumps({'type': 'error', 'content': e.message}, ensure_ascii=False)}\n\n"
            except Exception as e:
                import traceback
                traceback.print_exc()
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"
        
        response = Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                'Connection': 'keep-alive'
            }
        )
        response.headers['Content-Type'] = 'text/event-stream; charset=utf-8'
        return response
    
    except JobFitException as e:
        return jsonify({'error': e.message}), e.status_code
    except Exception as e:
        return jsonify({'error': '服务器内部错误'}), 500

@jobfit_bp.route('/upload/<file_type>', methods=['POST'])
def upload_file(file_type):
    """
    文件上传接口
    
    参数：
    - file_type: 'resume' 或 'jd'
    
    表单数据：
    - file: 文件
    - username: 用户名（可选）
    
    响应：
    {
        "path": "文件路径",
        "filename": "文件名",
        "message": "上传成功"
    }
    """
    try:
        if file_type not in ['resume', 'jd']:
            raise FileUploadException('无效的文件类型')
        
        if 'file' not in request.files:
            raise FileUploadException('没有文件')
        
        file = request.files['file']
        if file.filename == '':
            raise FileUploadException('文件名为空')
        
        # 确保上传目录存在
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        
        # 生成文件名
        username = request.form.get('username', 'anonymous')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{username}_{file_type}_{timestamp}_{file.filename}"
        filepath = os.path.join(settings.UPLOAD_DIR, filename)
        
        # 保存文件
        file.save(filepath)
        
        return jsonify({
            'path': filepath,
            'filename': filename,
            'message': '上传成功'
        }), 200
    
    except JobFitException as e:
        return jsonify({'error': e.message}), e.status_code
    except Exception as e:
        return jsonify({'error': '文件上传失败'}), 500

@jobfit_bp.route('/report/generate', methods=['POST'])
def generate_report():
    """
    生成报告接口（自动在匹配后调用）
    
    请求体：
    {
        "match_result": {...},
        "username": "用户名",
        "additional_advice": "额外建议"
    }
    
    响应：
    {
        "report_path": "报告文件路径",
        "message": "报告已生成"
    }
    """
    try:
        data = request.get_json()
        match_result = data.get('match_result', {})
        username = data.get('username', 'anonymous')
        additional_advice = data.get('additional_advice', '')
        
        report_path = jobfit_service.generate_report(match_result, username, additional_advice)
        
        return jsonify({
            'report_path': report_path,
            'message': '报告已生成'
        }), 200
    
    except JobFitException as e:
        return jsonify({'error': e.message}), e.status_code
    except Exception as e:
        return jsonify({'error': '报告生成失败'}), 500

@jobfit_bp.route('/report/list', methods=['GET'])
def get_report_list():
    """
    获取用户报告列表
    
    查询参数：
    - username: 用户名
    
    响应：
    {
        "reports": [
            {
                "filename": "报告文件名",
                "path": "报告路径",
                "created_time": "创建时间",
                "timestamp": "时间戳"
            }
        ]
    }
    """
    try:
        username = request.args.get('username', 'anonymous')
        reports = jobfit_service.get_user_reports(username)
        
        return jsonify({
            'reports': reports
        }), 200
    
    except JobFitException as e:
        return jsonify({'error': e.message}), e.status_code
    except Exception as e:
        return jsonify({'error': '获取报告列表失败'}), 500

@jobfit_bp.route('/report/content', methods=['GET'])
def get_report_content():
    """
    获取报告内容
    
    查询参数：
    - username: 用户名
    - filename: 报告文件名
    
    响应：
    {
        "content": "报告内容",
        "filename": "报告文件名"
    }
    """
    try:
        username = request.args.get('username', 'anonymous')
        filename = request.args.get('filename', '')
        
        if not filename:
            raise ValidationException('请提供报告文件名')
        
        content = jobfit_service.get_report_content(username, filename)
        
        if content is None:
            return jsonify({'error': '报告文件不存在'}), 404
        
        return jsonify({
            'content': content,
            'filename': filename
        }), 200
    
    except JobFitException as e:
        return jsonify({'error': e.message}), e.status_code
    except Exception as e:
        return jsonify({'error': '获取报告内容失败'}), 500


@jobfit_bp.route('/report/delete', methods=['DELETE'])
def delete_report():
    """
    删除指定用户的报告文件
    ---
    parameters:
      - name: username
        in: query
        type: string
        required: true
        description: 用户名
      - name: filename
        in: query
        type: string
        required: true
        description: 报告文件名
    responses:
      200:
        description: 删除成功
        schema:
          type: object
          properties:
            success:
              type: boolean
              description: 是否删除成功
            message:
              type: string
              description: 提示信息
      400:
        description: 参数错误
      404:
        description: 报告文件不存在
    """
    try:
        username = request.args.get('username', 'anonymous')
        filename = request.args.get('filename', '')
        
        if not filename:
            raise ValidationException('请提供报告文件名')
        
        # 构建报告文件路径
        reports_dir = os.path.join(os.getcwd(), 'reports', username)
        report_path = os.path.join(reports_dir, filename)
        
        if not os.path.exists(report_path):
            return jsonify({'success': False, 'message': '报告文件不存在'}), 404
        
        # 删除文件
        os.remove(report_path)
        
        return jsonify({
            'success': True,
            'message': '报告删除成功'
        }), 200
    
    except JobFitException as e:
        return jsonify({'success': False, 'message': e.message}), e.status_code
    except Exception as e:
        return jsonify({'success': False, 'message': '删除报告失败'}), 500
