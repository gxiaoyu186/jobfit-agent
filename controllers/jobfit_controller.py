"""
JobFit 控制器 - 处理匹配分析相关的 HTTP 请求
特性：
1. 仅负责处理 HTTP 请求和响应
2. 支持流式响应
3. 统一异常处理和响应格式
"""

import json
from flask import Blueprint, request, jsonify, Response
from services.jobfit_service import JobFitService
from config.settings import settings
from exceptions.jobfit_exceptions import JobFitException, FileUploadException
from datetime import datetime
import os

# 创建蓝图
jobfit_bp = Blueprint('jobfit', __name__, url_prefix='/api/v1')

# 初始化服务
jobfit_service = JobFitService()

@jobfit_bp.route('/agent/chat', methods=['POST'])
def agent_chat():
    """
    Agent 对话接口（流式响应）
    
    请求体：
    {
        "message": "用户消息",
        "resume_path": "简历文件路径（可选）",
        "jd_path": "JD文件路径（可选）"
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
        
        def generate():
            try:
                # 调用服务层进行分析
                for chunk in jobfit_service.analyze_match(
                    resume_path=resume_path,
                    jd_path=jd_path,
                    message=message
                ):
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            except JobFitException as e:
                yield f"data: {json.dumps({'type': 'error', 'content': e.message}, ensure_ascii=False)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"
        
        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )
    
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
    生成报告接口
    
    请求体：
    {
        "match_result": {...},
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
        additional_advice = data.get('additional_advice', '')
        
        report_path = jobfit_service.generate_report(match_result, additional_advice)
        
        return jsonify({
            'report_path': report_path,
            'message': '报告已生成'
        }), 200
    
    except JobFitException as e:
        return jsonify({'error': e.message}), e.status_code
    except Exception as e:
        return jsonify({'error': '报告生成失败'}), 500
