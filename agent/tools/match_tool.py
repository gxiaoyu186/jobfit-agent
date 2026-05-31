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
    match_score: float
    required_skills_match: List[str]
    required_skills_missing: List[str]
    preferred_skills_match: List[str]
    preferred_skills_missing: List[str]
    recommendations: List[str]

def _normalize_field_names(data: dict) -> dict:
    """将中文字段名映射到英文字段名"""
    field_mapping = {
        "匹配得分": "match_score",
        "匹配分数": "match_score",
        "得分": "match_score",
        "必需技能匹配": "required_skills_match",
        "已匹配必需技能": "required_skills_match",
        "必需技能已匹配": "required_skills_match",
        "缺失必需技能": "required_skills_missing",
        "必需技能缺失": "required_skills_missing",
        "加分技能匹配": "preferred_skills_match",
        "已匹配加分技能": "preferred_skills_match",
        "加分技能已匹配": "preferred_skills_match",
        "缺失加分技能": "preferred_skills_missing",
        "加分技能缺失": "preferred_skills_missing",
        "建议": "recommendations",
        "学习建议": "recommendations",
        "改进建议": "recommendations"
    }

    normalized = {}
    for key, value in data.items():
        new_key = field_mapping.get(key, key)
        if isinstance(value, list):
            normalized[new_key] = value
        elif isinstance(value, dict):
            normalized[new_key] = _normalize_field_names(value)
        else:
            normalized[new_key] = value
    return normalized

@tool
def match_resume_to_jd(resume_text: str, jd_text: str) -> str:
    """
    对比简历和职位描述，返回权重化匹配分析 JSON。

    Args:
        resume_text: 简历文本内容
        jd_text: 职位描述文本内容

    Returns:
        JSON 格式的匹配分析结果
    """
    import time
    start_time = time.time()
    
    llm = _get_llm()

    prompt = f"""你是招聘专家。分析简历和JD，输出JSON。

必须字段（英文）：match_score, required_skills_match, required_skills_missing, preferred_skills_match, preferred_skills_missing, recommendations

简历：{resume_text[:2000]}

JD：{jd_text[:2000]}

输出纯JSON："""

    max_retries = 1
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            content = response.content.strip()

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            content = content.strip()
            result = json.loads(content)
            result = _normalize_field_names(result)
            validated = MatchResult(**result)
            return json.dumps(validated.dict(), ensure_ascii=False)

        except (json.JSONDecodeError, ValidationError) as e:
            pass

    return json.dumps({
        "match_score": 0,
        "required_skills_match": [],
        "required_skills_missing": [],
        "preferred_skills_match": [],
        "preferred_skills_missing": [],
        "recommendations": ["匹配分析服务暂时不可用"]
    }, ensure_ascii=False)
