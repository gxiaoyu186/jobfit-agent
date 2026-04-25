from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.agents import create_agent
import sqlite3
import os
from tool import *

load_dotenv()


def create_jobfit_agent():
    # 初始化多模态模型（用于决策和工具调用）
    model = init_chat_model(
        model="qwen3.5-plus",
        model_provider="openai",
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("API_KEY")
    )

    # 工具列表
    tools = [
        extract_text_from_image,
        match_resume_to_jd,
        search_internet,
        suggest_learning,
        reflect_on_match,
        generate_report
    ]

    # 系统提示词 —— 引导 Agent 按正确流程工作
    system_prompt = """
    你是一个求职教练 Agent，必须**严格按流程执行**，不能跳过任何工具调用。

    ## 工具
    1. `extract_text_from_image(image_path)` - 从图片提取文字
    2. `match_resume_to_jd(resume_text, jd_text)` - 返回加权匹配分数、必需/加分技能的匹配与缺失、建议
    3. `reflect_on_match(match_json_str)` - 反思匹配结果，判断是否需要搜索
    4. `search_internet(query)` - 搜索面经/学习资料
    5. `suggest_learning(missing_skills)` - 生成学习路径
    6. `generate_report(match_json_str, additional_advice)` - 生成 Markdown 报告

    ## 强制工作流（当用户提供两张图片路径并要求匹配时）
    1. **必须**连续两次调用 `extract_text_from_image`，分别提取简历和 JD 的文字。
    2. **必须**调用 `match_resume_to_jd`，传入上一步得到的文本。
    3. **必须**调用 `reflect_on_match`，传入匹配结果 JSON。
       - 如果反思结果中包含“需要搜索”，则必须调用 `search_internet`（用缺失技能作为关键词）。
    4. **必须**将匹配分数、匹配/缺失技能、反思结论、搜索摘要（如有）组织成最终回复。
    5. 如果用户要求“生成报告”，**必须**调用 `generate_report`。

    ## 其他规则
    - **替换/更新**：用户明确说“替换简历/JD”时，只使用最新提交的图片或文本，忽略之前的内容。
    - **主动追问**：在首次匹配回复后，可以主动询问一项额外信息（如项目经验、求职时限），但每次最多追问一次。
    - **单张图片**：如果只提供一张图片，询问另一张。
    - **禁止编造**：绝对不能在不调用工具的情况下回复“无法提取”或类似的错误提示。工具返回的错误应原样告知用户。

    ## 示例用户输入
    > 分析匹配度。简历路径：/a/resume.png，JD路径：/b/jd.png

    你必须按上述流程逐步调用工具，最终输出分析结果。
    """

    # 检查点持久化
    conn = sqlite3.connect("resources/jobfit_agent.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer
    )
    return agent


def stream_jobfit_agent(agent, user_input, config):
    import time
    print(f"[{time.strftime('%H:%M:%S')}] 开始stream...")
    for chunk in agent.stream({"messages": [HumanMessage(content=user_input)]}, config):
        print(f"[{time.strftime('%H:%M:%S')}] 收到chunk，keys: {chunk.keys() if hasattr(chunk, 'keys') else type(chunk)}")
        if "agent" in chunk:
            for msg in chunk["agent"].get("messages", []):
                if hasattr(msg, "content") and msg.content:
                    print(f"[{time.strftime('%H:%M:%S')}] yield agent: {str(msg.content)[:100]}...")
                    yield {"type": "agent", "content": msg.content}
        elif "tools" in chunk:
            for msg in chunk["tools"].get("messages", []):
                if hasattr(msg, "content") and msg.content:
                    print(f"[{time.strftime('%H:%M:%S')}] yield tool: {str(msg.content)[:100]}...")
                    yield {"type": "tool", "content": msg.content}
        elif "model" in chunk:
            for msg in chunk["model"].get("messages", []):
                if hasattr(msg, "content") and msg.content:
                    print(f"[{time.strftime('%H:%M:%S')}] yield model: {str(msg.content)[:100]}...")
                    yield {"type": "agent", "content": msg.content}
        elif "__end__" in chunk:
            for msg in chunk["__end__"].get("messages", []):
                if hasattr(msg, "content") and msg.content:
                    print(f"[{time.strftime('%H:%M:%S')}] yield end agent: {str(msg.content)[:100]}...")
                    yield {"type": "agent", "content": msg.content}
        else:
            print(f"[{time.strftime('%H:%M:%S')}] 未处理的chunk结构: {str(chunk)[:200]}")
    print(f"[{time.strftime('%H:%M:%S')}] stream结束")