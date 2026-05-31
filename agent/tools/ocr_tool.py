"""
OCR 工具 - 从图片中提取文字
"""

import os
import base64
import time
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from config.settings import settings

# 模型缓存
_vision_model = None

def _get_vision_model():
    """获取缓存的多模态模型"""
    global _vision_model
    if _vision_model is None:
        _vision_model = ChatOpenAI(
            model=settings.MODEL_NAME,
            base_url=settings.BASE_URL,
            api_key=settings.API_KEY,
            temperature=settings.MODEL_TEMPERATURE,
            timeout=180  # 增加到3分钟
        )
    return _vision_model

@tool
def extract_text_from_image(image_path: str) -> str:
    """
    从图片中提取文字（支持本地路径或 URL）。
    
    Args:
        image_path: 图片文件路径或 http/https 链接。
        
    Returns:
        图片中的文字内容（纯文本）。
    """
    llm = _get_vision_model()
    
    if image_path.startswith(("http://", "https://")):
        image_url = image_path
    else:
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
            mime = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
            image_url = f"data:{mime};base64,{data}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "请提取这张图片中的所有文字，只返回文字内容，不要加任何解释。"}
            ]
        }
    ]
    
    # 只重试1次，避免重复调用浪费金钱
    max_retries = 1
    
    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)
            return response.content
        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            else:
                raise Exception(f"OCR 提取失败: {error_msg}")
    
    return ""