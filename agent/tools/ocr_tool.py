"""
OCR 工具 - 从图片中提取文字
特性：
1. 支持本地图片和网络图片
2. 使用多模态模型进行文字识别
3. 模型实例缓存，避免重复初始化
"""

import os
import base64
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
            temperature=settings.MODEL_TEMPERATURE
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
    # 使用缓存的多模态模型
    vision_model = _get_vision_model()
    
    # 处理图片输入
    if image_path.startswith(("http://", "https://")):
        image_url = image_path
    else:
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
            mime = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
            image_url = f"data:{mime};base64,{data}"

    # 标准 OpenAI 多模态消息格式
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "请提取这张图片中的所有文字，只返回文字内容，不要加任何解释。"}
            ]
        }
    ]
    
    response = vision_model.invoke(messages)
    return response.content
