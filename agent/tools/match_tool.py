"""
匹配工具 - 对比简历与 JD 的匹配度
特性：
1. 识别必需技能和加分技能
2. 加权评分算法
3. 返回结构化结果
"""

import json
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, ValidationError
from typing import List
from config.settings import settings

# 模型缓存
_llm_model = None

def _get_llm():
    """获取缓存的 LLM 模型"""
    global _llm_model
    if _llm_model is None:
        _llm_model = init_chat_model(
            model=settings.MODEL_NAME,
            model_provider="openai",
            base_url=settings.BASE_URL,
            api_key=settings.API_KEY
        )
    return _llm_model

class MatchResult(BaseModel):
    """匹配结果数据模型"""
    match_score: int
    required_skills_match: List[str]
    required_skills_missing: List[str]
    preferred_skills_match: List[str]
    preferred_skills_missing: List[str]
    recommendations: List[str]

@tool
def match_resume_to_jd(resume_text: str, jd_text: str) -> str:
    """
    对比简历和职位描述，返回权重化匹配分析 JSON。
    
    Args:
        resume_text: 简历文本内容
        jd_text: 职位描述文本内容
        
    Returns:
        JSON 格式的匹配分析结果，包含：
        - match_score: 匹配分数 (0-100)
        - required_skills_match: 匹配的必需技能
        - required_skills_missing: 缺失的必需技能
        - preferred_skills_match: 匹配的加分技能
        - preferred_skills_missing: 缺失的加分技能
        - recommendations: 改进建议
    """
    llm = _get_llm()
    
    prompt = f"""
        你是一名资深招聘专家。请分析以下简历和职位描述（JD），识别 JD 中的"必需技能"和"加分技能"，然后与简历对比，输出 JSON。
        
        **步骤：**
        1. 从 JD 中提取所有技能要求，分类为：
           - required: 明确写"必须"、"精通"、"3年以上经验"等强制性词汇的技能
           - preferred: 写"加分"、"熟悉"、"有经验优先"等非强制性技能
        2. 对照简历，判断简历中是否提到这些技能（同义词、相关技术也算匹配）。
        3. 计算 match_score（0-100）：
           - 必需技能权重 70%，每缺失一项扣 (70 / 总必需项数)
           - 加分技能权重 30%，每匹配一项加 (30 / 总加分项数)，缺失不扣分
           - 基础分从 0 开始，加上必需匹配得分再加加分匹配得分
        4. 给出具体 recommendations（针对缺失的必需技能，提出最紧急的中文学习建议）。
        
        【强制要求】所有输出内容必须使用简体中文，包括 JSON 中的字符串、建议内容等，严禁出现中英文混杂。
        
        简历内容：
        {resume_text}
        
        职位描述：
        {jd_text}
        
        输出 JSON，不要其他文字：
        {{
            "match_score": 整数,
            "required_skills_match": ["技能1", "技能2"],
            "required_skills_missing": ["技能3"],
            "preferred_skills_match": ["技能4"],
            "preferred_skills_missing": ["技能5"],
            "recommendations": ["建议1", "建议2"]
        }}
        """
    
    response = llm.invoke(prompt)
    content = response.content.strip()
    
    # 清理 JSON 格式
    if content.startswith("```json"):
        content = content[7:]
    if content.endswith("```"):
        content = content[:-3]
    
    # 解析和验证
    try:
        result = json.loads(content)
        validated = MatchResult(**result)
        result = validated.dict()
    except json.JSONDecodeError:
        result = {
            "match_score": 0,
            "required_skills_match": [],
            "required_skills_missing": [],
            "preferred_skills_match": [],
            "preferred_skills_missing": [],
            "recommendations": ["JSON解析失败，请检查输入"]
        }
    except ValidationError as e:
        result = {
            "match_score": 0,
            "required_skills_match": [],
            "required_skills_missing": [],
            "preferred_skills_match": [],
            "preferred_skills_missing": [],
            "recommendations": [f"结构化验证失败: {e}"]
        }
    
    return json.dumps(result, ensure_ascii=False)
