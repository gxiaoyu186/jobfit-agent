"""
JobFit 服务 - 处理简历与JD匹配的核心业务逻辑
特性：
1. 封装匹配分析的业务流程
2. 与 Agent 层解耦
3. 支持流式响应
4. 统一异常处理
5. 报告存储管理（按用户存储，最多10个）
"""

import json
import os
from datetime import datetime
from typing import Generator, Optional, List, Dict

from config.settings import settings
from agent.jobfit_agent import create_jobfit_agent, stream_jobfit_agent
from exceptions.jobfit_exceptions import ValidationException, ModelException

class AnalysisResult:
    """分析结果数据类"""
    
    def __init__(self, match_score: int, required_skills_match: list, 
                 required_skills_missing: list, preferred_skills_match: list,
                 preferred_skills_missing: list, recommendations: list):
        self.match_score = match_score
        self.required_skills_match = required_skills_match
        self.required_skills_missing = required_skills_missing
        self.preferred_skills_match = preferred_skills_match
        self.preferred_skills_missing = preferred_skills_missing
        self.recommendations = recommendations
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            'match_score': self.match_score,
            'required_skills_match': self.required_skills_match,
            'required_skills_missing': self.required_skills_missing,
            'preferred_skills_match': self.preferred_skills_match,
            'preferred_skills_missing': self.preferred_skills_missing,
            'recommendations': self.recommendations
        }

class JobFitService:
    """JobFit 核心服务类"""
    
    def __init__(self):
        """初始化 Agent 实例（懒加载）"""
        self._agent = None
    
    @property
    def agent(self):
        """懒加载 Agent 实例"""
        if self._agent is None:
            self._agent = create_jobfit_agent()
        return self._agent
    
    def analyze_match(self, resume_path: str = "", jd_path: str = "", 
                      message: str = "分析匹配度", username: str = "anonymous",
                      auto_generate_report: bool = True) -> Generator[dict, None, None]:
        """
        分析简历与 JD 的匹配度（流式响应）
        
        用户可以上传简历和JD图片，同时可以在消息中输入额外的文字说明，
        系统会综合图片识别结果和用户消息进行分析。
        
        Args:
            resume_path: 简历文件路径
            jd_path: JD 文件路径
            message: 用户消息（可包含文字形式的简历和JD内容）
            username: 用户名（用于自动生成报告）
            auto_generate_report: 是否自动生成报告
        
        Yields:
            流式响应数据块
            
        Raises:
            ValidationException: 参数验证失败
            ModelException: 模型调用失败
        """
        import re
        
        # 参数验证 - 至少需要图片路径或消息内容
        has_image_input = resume_path and jd_path
        
        if not message and not has_image_input:
            raise ValidationException('请上传简历和JD图片，或在输入框中输入内容')
        
        # 验证图片路径（如果提供了）
        if resume_path and not os.path.exists(resume_path):
            raise ValidationException(f'简历文件不存在: {resume_path}')
        
        if jd_path and not os.path.exists(jd_path):
            raise ValidationException(f'JD文件不存在: {jd_path}')
        
        # 构建用户输入
        if has_image_input:
            # 有图片：使用路径进行OCR，同时结合用户消息
            user_input = f"""{message}。

简历路径: {resume_path}
JD路径: {jd_path}

请分析以上内容的匹配度。"""
        else:
            # 无图片：直接使用用户消息进行分析
            user_input = f"""{message}

请分析以上内容的匹配度。"""
        
        # 配置会话
        config = {"configurable": {"thread_id": "jobfit_user_session"}}
        
        # 用于存储匹配结果
        match_result = {}
        
        try:
            # 流式调用 Agent
            for chunk in stream_jobfit_agent(self.agent, user_input, config):
                yield chunk
                
                chunk_type = chunk.get('type')
                content = chunk.get('content', '')
                
                if 'match_score' in content:
                    try:
                        json_match = re.search(r'\{[\s\S]*\}', content)
                        if json_match:
                            json_str = json_match.group()
                            match_result = json.loads(json_str)
                    except Exception:
                        pass
            
            yield {'type': 'done', 'content': ''}
            
            if auto_generate_report and match_result and isinstance(match_result, dict) and 'match_score' in match_result:
                try:
                    self.generate_report(match_result, username)
                    yield {'type': 'report_generated', 'content': '报告已自动生成'}
                except Exception:
                    pass
            
        except Exception as e:
            raise ModelException(f'分析失败: {str(e)}')
    
    def generate_report(self, match_result: dict, username: str = "anonymous", additional_advice: str = "") -> str:
        """
        生成匹配分析报告并按用户存储
        
        Args:
            match_result: 匹配结果字典
            username: 用户名（用于按用户存储）
            additional_advice: 额外建议
            
        Returns:
            报告文件路径
        """
        # 创建用户报告目录
        user_report_dir = os.path.join(settings.REPORTS_DIR, username)
        os.makedirs(user_report_dir, exist_ok=True)
        
        # 检查并清理超出数量限制的报告（最多10个）
        self._cleanup_old_reports(user_report_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"jobfit_report_{timestamp}.md"
        report_path = os.path.join(user_report_dir, report_filename)
        
        # 提取并清理学习建议
        recommendations = match_result.get('recommendations', [])
        if isinstance(recommendations, list):
            recommendations = [rec.strip() for rec in recommendations if rec.strip()]
            # 清理建议中可能存在的 Markdown 标题符号
            recommendations = [rec.lstrip('#').strip() for rec in recommendations]
        else:
            recommendations = []
        
        # 清理补充建议中的 Markdown 标题符号
        additional_advice_clean = additional_advice.strip() if additional_advice else '无'
        if additional_advice_clean and additional_advice_clean != '无':
            additional_advice_clean = additional_advice_clean.lstrip('#').strip()
        
        content = f"""# JobFit 职业匹配分析报告

生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 综合匹配分数：{match_result.get('match_score', 'N/A')} / 100

## 必需技能匹配情况
- ✅ 已匹配：{', '.join(match_result.get('required_skills_match', [])) or '无'}
- ❌ 缺失：{', '.join(match_result.get('required_skills_missing', [])) or '无'}

## 加分技能情况
- ✅ 已匹配：{', '.join(match_result.get('preferred_skills_match', [])) or '无'}
- ❌ 缺失：{', '.join(match_result.get('preferred_skills_missing', [])) or '无'}

## 学习建议
{chr(10).join(f'- {rec}' for rec in recommendations) if recommendations else '无'}

## 补充建议
{additional_advice_clean}
"""
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        return report_path
    
    def _cleanup_old_reports(self, user_report_dir: str, max_reports: int = 10):
        """
        清理用户超出数量限制的旧报告
        
        Args:
            user_report_dir: 用户报告目录
            max_reports: 最大报告数量（默认10个）
        """
        if not os.path.exists(user_report_dir):
            return
        
        # 获取所有报告文件并按修改时间排序
        report_files = []
        for filename in os.listdir(user_report_dir):
            if filename.startswith('jobfit_report_') and filename.endswith('.md'):
                filepath = os.path.join(user_report_dir, filename)
                if os.path.isfile(filepath):
                    report_files.append((filepath, os.path.getmtime(filepath)))
        
        # 按修改时间升序排序（最旧的在前）
        report_files.sort(key=lambda x: x[1])
        
        # 删除超出限制的旧报告
        while len(report_files) > max_reports:
            oldest_file = report_files.pop(0)
            os.remove(oldest_file[0])
    
    def get_user_reports(self, username: str) -> List[Dict]:
        """
        获取用户的报告列表
        
        Args:
            username: 用户名
            
        Returns:
            报告列表，包含文件名、路径和创建时间
        """
        user_report_dir = os.path.join(settings.REPORTS_DIR, username)
        reports = []
        
        if os.path.exists(user_report_dir):
            for filename in os.listdir(user_report_dir):
                if filename.startswith('jobfit_report_') and filename.endswith('.md'):
                    filepath = os.path.join(user_report_dir, filename)
                    if os.path.isfile(filepath):
                        # 从文件名提取时间戳
                        timestamp_str = filename.replace('jobfit_report_', '').replace('.md', '')
                        try:
                            created_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        except:
                            created_time = datetime.fromtimestamp(os.path.getctime(filepath))
                        
                        reports.append({
                            'filename': filename,
                            'path': filepath,
                            'created_time': created_time.strftime("%Y-%m-%d %H:%M:%S"),
                            'timestamp': timestamp_str
                        })
        
        # 按创建时间降序排序（最新的在前）
        reports.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return reports
    
    def get_report_content(self, username: str, filename: str) -> Optional[str]:
        """
        获取报告内容
        
        Args:
            username: 用户名
            filename: 报告文件名
            
        Returns:
            报告内容，如果文件不存在返回 None
        """
        user_report_dir = os.path.join(settings.REPORTS_DIR, username)
        filepath = os.path.join(user_report_dir, filename)
        
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        
        return None
