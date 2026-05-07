"""
JobFit 服务 - 处理简历与JD匹配的核心业务逻辑
特性：
1. 封装匹配分析的业务流程
2. 与 Agent 层解耦
3. 支持流式响应
4. 统一异常处理
"""

import json
import os
from datetime import datetime
from typing import Generator, Optional

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
                      message: str = "分析匹配度") -> Generator[dict, None, None]:
        """
        分析简历与 JD 的匹配度（流式响应）
        
        用户可以上传简历和JD图片，同时可以在消息中输入额外的文字说明，
        系统会综合图片识别结果和用户消息进行分析。
        
        Args:
            resume_path: 简历文件路径
            jd_path: JD 文件路径
            message: 用户消息（可包含文字形式的简历和JD内容）
        
        Yields:
            流式响应数据块
            
        Raises:
            ValidationException: 参数验证失败
            ModelException: 模型调用失败
        """
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
        
        try:
            # 流式调用 Agent
            for chunk in stream_jobfit_agent(self.agent, user_input, config):
                yield chunk
            
            # 标记完成
            yield {'type': 'done', 'content': ''}
            
        except Exception as e:
            raise ModelException(f'分析失败: {str(e)}')
    
    def generate_report(self, match_result: dict, additional_advice: str = "") -> str:
        """
        生成匹配分析报告
        
        Args:
            match_result: 匹配结果字典
            additional_advice: 额外建议
            
        Returns:
            报告文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(settings.REPORTS_DIR, f"jobfit_report_{timestamp}.md")
        
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
{chr(10).join(f'- {rec}' for rec in match_result.get('recommendations', []))}

## 补充建议
{additional_advice if additional_advice else '无'}
"""
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        return report_path
