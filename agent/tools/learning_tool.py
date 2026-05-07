"""
学习建议工具 - 根据缺失技能提供学习路径
特性：
1. 解析缺失技能列表
2. 生成针对性学习建议
3. 推荐学习资源和实践项目
"""

import json
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
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

@tool
def suggest_learning(missing_skills: str) -> str:
    """
    根据缺失的技能列表，提供学习路径和建议。
    
    Args:
        missing_skills: 缺失技能列表，格式可以是 "技能A, 技能B" 或 JSON 数组字符串。
        
    Returns:
        学习建议文本。
    """
    llm = _get_llm()
    
    # 解析输入
    try:
        if missing_skills.strip().startswith("["):
            skills_list = json.loads(missing_skills)
        else:
            skills_list = [s.strip() for s in missing_skills.split(",")]
    except:
        skills_list = [missing_skills]
    
    skill_str = ", ".join(skills_list)
    
    prompt = f"""
        用户需要学习以下技能：{skill_str}。
        请提供一份简短的学习路径建议，包括推荐的学习顺序、免费资源（如网站、课程）、以及实践项目建议。输出纯文本，不超过 200 字。
        """
    
    response = llm.invoke(prompt)
    return response.content
