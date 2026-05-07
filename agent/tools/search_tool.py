"""
搜索工具 - 在互联网上搜索相关信息
特性：
1. 使用 Tavily 搜索引擎
2. 搜索结果缓存
3. 返回结构化摘要
"""

from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from config.settings import settings

# 搜索工具缓存
_tavily_search = None

def _get_tavily_search():
    """获取缓存的搜索工具"""
    global _tavily_search
    if _tavily_search is None:
        _tavily_search = TavilySearch(
            max_results=3,
            topic="general",
            api_key=settings.TAVILY_API_KEY
        )
    return _tavily_search

@tool
def search_internet(query: str) -> str:
    """
    搜索互联网获取相关面经、学习资源或岗位信息。
    
    Args:
        query: 搜索关键词（例如 "Python 学习资源"）。
        
    Returns:
        搜索结果的文本摘要。
    """
    # 使用缓存的搜索工具
    tavily = _get_tavily_search()
    
    try:
        result = tavily.run(query)
        if isinstance(result, dict):
            snippets = [item.get("snippet", "") for item in result.get("results", [])]
            return "\n".join(snippets)
        else:
            return str(result)
    except Exception as e:
        return f"搜索失败: {e}。请检查 TAVILY_API_KEY 是否配置正确。"
