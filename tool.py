import os
import base64
from io import BytesIO
from PIL import Image
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
import json
from datetime import datetime
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from typing import List

load_dotenv()

class MatchResult(BaseModel):
    match_score: int
    required_skills_match: List[str]
    required_skills_missing: List[str]
    preferred_skills_match: List[str]
    preferred_skills_missing: List[str]
    recommendations: List[str]

# 定义model，同时定义调用方法，避免重复调用--------------

model = None

def get_llm():
    global model
    if model is None:
        from dotenv import load_dotenv
        load_dotenv()
        model = init_chat_model(
            model="qwen3.5-plus",
            model_provider="openai",
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("API_KEY")
        )
    return model

def _encode_image(image_path: str) -> str:
    """将本地图片转为 base64 data url"""
    with open(image_path, "rb") as f:
        img_data = f.read()
    # 尝试推断图片格式
    try:
        img = Image.open(BytesIO(img_data))
        format = img.format.lower()
    except:
        format = "jpeg"
    base64_str = base64.b64encode(img_data).decode("utf-8")
    return f"data:image/{format};base64,{base64_str}"

# 用于提取图片中的文字，是匹配的前置工作————


@tool
def extract_text_from_image(image_path: str) -> str:
    """
    从图片中提取文字（支持本地路径或 URL）。
    输入: 图片文件路径或 http/https 链接。
    输出: 图片中的文字内容（纯文本）。
    """

    # 1. 明确使用 qwen3.5-plus 这个多模态模型
    vision_model = ChatOpenAI(
        model="qwen3.5-plus",
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("API_KEY"),
        temperature=0.0
    )
    # 2. 处理图片输入 (本地文件转 base64，网络图片直接用 URL)
    if image_path.startswith(("http://", "https://")):
        image_url = image_path
    else:
        with open(image_path, "rb") as f:
            import base64
            data = base64.b64encode(f.read()).decode()
            mime = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
            image_url = f"data:{mime};base64,{data}"

    # 3. 标准 OpenAI 多模态消息格式
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

# 用于将简历与岗位jd进行匹配 ————————————
@tool
def match_resume_to_jd(resume_text: str, jd_text: str) -> str:
    """
    对比简历和职位描述，返回权重化匹配分析 JSON。
    输出格式：
    {
        "match_score": 75,
        "required_skills_match": ["Python"],
        "required_skills_missing": ["Docker"],
        "preferred_skills_match": ["SQL"],
        "preferred_skills_missing": ["K8s"],
        "recommendations": ["重点学习 Docker，参加 CKA 认证"]
    }
    """
    llm = get_llm()
    prompt = f"""
        你是一名资深招聘专家。请分析以下简历和职位描述（JD），识别 JD 中的“必需技能”和“加分技能”，然后与简历对比，输出 JSON。
        
        **步骤：**
        1. 从 JD 中提取所有技能要求，分类为：
           - required: 明确写“必须”、“精通”、“3年以上经验”等强制性词汇的技能
           - preferred: 写“加分”、“熟悉”、“有经验优先”等非强制性技能
        2. 对照简历，判断简历中是否提到这些技能（同义词、相关技术也算匹配）。
        3. 计算 match_score（0-100）：
           - 必需技能权重 70%，每缺失一项扣 (70 / 总必需项数)
           - 加分技能权重 30%，每匹配一项加 (30 / 总加分项数)，缺失不扣分
           - 基础分从 0 开始，加上必需匹配得分再加加分匹配得分
        4. 给出具体 recommendations（针对缺失的必需技能，提出最紧急的学习建议）。
        
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
    # 解析 JSON 的容错代码（与之前类似）
    content = response.content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.endswith("```"):
        content = content[:-3]
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

# 用于在网络上搜寻技能相关知识 ————————————————————
@tool
def search_internet(query: str) -> str:
    """
    搜索互联网获取相关面经、学习资源或岗位信息。
    输入: 搜索关键词（例如 "Python 学习资源"）。
    输出: 搜索结果的文本摘要。
    """
    tavily = TavilySearch(
        max_results=3,
        topic="general",
        api_key=os.getenv("TAVILY_API_KEY")
    )
    try:
        result = tavily.run(query)
        if isinstance(result, dict):
            snippets = [item.get("snippet", "") for item in result.get("results", [])]
            return "\n".join(snippets)
        else:
            return str(result)
    except Exception as e:
        return f"搜索失败: {e}。请检查 TAVILY_API_KEY 是否配置正确。"

# 给出学习建议————————————
@tool
def suggest_learning(missing_skills: str) -> str:
    """
    根据缺失的技能列表，提供学习路径和建议。
    输入: 缺失技能列表，格式可以是 "技能A, 技能B" 或 JSON 数组字符串。
    输出: 学习建议文本。
    """
    llm = get_llm()
    # 尝试解析输入
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

# 自我反思————————
@tool
def reflect_on_match(match_json_str: str, resume_text: str = "", jd_text: str = "") -> str:
    """
    对匹配结果进行自我反思，评估是否合理，给出改进建议。
    输入：match_json_str（方向1输出的JSON字符串），可选简历和JD原文。
    输出：反思文本（例如：“匹配分数较低，建议重点补充缺失的必需技能 Docker。是否需要我帮你搜索学习资源？”）
    """
    llm = get_llm()
    prompt = f"""
        你是一个自我批判的职业教练。现在有一个匹配分析结果如下：
        {match_json_str}
        
        请你反思：
        1. 这个匹配分析是否合理？有没有遗漏关键点？
        2. 如果匹配分数低于60分，指出最重要的一个改进方向。
        3. 是否需要额外搜索互联网来获取学习建议？输出“需要搜索”或“不需要”。
        输出纯文本，简洁明了。
        """
    response = llm.invoke(prompt)
    return response.content

# 生成匹配报告——————————
@tool
def generate_report(match_json_str: str, additional_advice: str = "") -> str:
    """
    根据匹配结果生成一份 Markdown 格式的报告，并保存到 reports/ 目录。
    输入：match_json_str（方向1的输出），additional_advice（额外建议，如搜索摘要）。
    输出：报告的文件路径。
    """
    data = json.loads(match_json_str)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("reports", exist_ok=True)
    report_path = f"reports/jobfit_report_{timestamp}.md"

    content = f"""# JobFit 职业匹配分析报告
        生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        
        ## 综合匹配分数：{data.get('match_score', 'N/A')} / 100
        
        ## 必需技能匹配情况
        - ✅ 已匹配：{', '.join(data.get('required_skills_match', [])) or '无'}
        - ❌ 缺失：{', '.join(data.get('required_skills_missing', [])) or '无'}
        
        ## 加分技能情况
        - ✅ 已匹配：{', '.join(data.get('preferred_skills_match', [])) or '无'}
        - ❌ 缺失：{', '.join(data.get('preferred_skills_missing', [])) or '无'}
        
        ## 学习建议
        {chr(10).join(f'- {rec}' for rec in data.get('recommendations', []))}
        
        ## 补充建议
        {additional_advice if additional_advice else '无'}
        """
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"报告已生成：{report_path}"