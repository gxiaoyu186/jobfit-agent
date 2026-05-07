"""
反思工具 - 对匹配结果进行自我反思
特性：
1. 评估匹配分析的合理性
2. 判断是否需要进一步搜索
3. 提供改进建议
"""

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
def reflect_on_match(match_json_str: str, resume_text: str = "", jd_text: str = "") -> str:
    """
    对匹配结果进行自我反思，评估是否合理，给出改进建议。
    
    Args:
        match_json_str: 匹配结果的 JSON 字符串
        resume_text: 简历原文（可选）
        jd_text: JD 原文（可选）
        
    Returns:
        反思文本，例如："匹配分数较低，建议重点补充缺失的必需技能 Docker。是否需要我帮你搜索学习资源？"
    """
    llm = _get_llm()
    
    prompt = f"""
        你是一个自我批判的职业教练。现在有一个匹配分析结果如下：
        {match_json_str}
        
        请你反思：
        1. 这个匹配分析是否合理？有没有遗漏关键点？
        2. 如果匹配分数低于60分，指出最重要的一个改进方向。
        3. 是否需要额外搜索互联网来获取学习建议？输出"需要搜索"或"不需要"。
        输出纯文本，简洁明了。
        """
    
    response = llm.invoke(prompt)
    return response.content
