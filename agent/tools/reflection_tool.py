"""
反思工具 - 对匹配结果进行自我反思
特性：
1. 评估匹配分析的合理性
2. 判断是否需要进一步搜索
3. 提供改进建议
"""

import time
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
def reflect_on_match(match_json_str: str) -> str:
    """
    对匹配结果进行自我反思，评估是否合理，给出改进建议。
    
    Args:
        match_json_str: 匹配结果的 JSON 字符串
        
    Returns:
        反思文本
    """
    start = time.time()
    
    llm = _get_llm()
    
    prompt = f"""你是职业教练。反思匹配结果：{match_json_str}
    
1. 分析是否合理？
2. 分数<60时指出最重要改进方向（中文）？
3. 输出"需要搜索"或"不需要"。

输出纯文本："""
    
    response = llm.invoke(prompt)
    return response.content
