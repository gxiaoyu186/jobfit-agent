"""
JobFit Agent 核心模块 - 定义 Agent 工作流和工具调用
"""

import re
import json
import threading
import time
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

from config.settings import settings
from agent.tools.ocr_tool import extract_text_from_image
from agent.tools.match_tool import match_resume_to_jd
from agent.tools.search_tool import search_internet
from agent.tools.learning_tool import suggest_learning
from agent.tools.reflection_tool import reflect_on_match

load_dotenv()

def create_jobfit_agent():
    model = init_chat_model(
        model=settings.MODEL_NAME,
        model_provider="openai",
        base_url=settings.BASE_URL,
        api_key=settings.API_KEY
    )

    tools = [
        extract_text_from_image,
        match_resume_to_jd,
        search_internet,
        suggest_learning,
        reflect_on_match
    ]

    system_prompt = """你是一个专业的求职教练，擅长分析简历与职位描述的匹配度。"""

    memory = MemorySaver()
    
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=memory
    )
    return agent, model


def stream_jobfit_agent(agent, model, resume_path, jd_path, user_input, config, has_image_input=True):
    def run_tool(tool_func, args):
        try:
            return ('success', tool_func.invoke(args))
        except Exception as e:
            return ('error', str(e))

    def stream_llm(prompt_text):
        for chunk in model.stream([HumanMessage(content=prompt_text)]):
            if hasattr(chunk, 'content') and chunk.content:
                yield {'type': 'reasoning', 'content': chunk.content}

    def yield_stage(stage_id, stage_name, icon, order, total):
        yield {
            'type': 'thinking_stage',
            'content': {'id': stage_id, 'name': stage_name, 'icon': icon, 'order': order},
            'message': f'阶段 {order}/{total}：{stage_name}'
        }

    try:
        TOTAL_STAGES = 6 if has_image_input else 5
        stage_num = 0

        if has_image_input:
            stage_num += 1
            yield from yield_stage('ocr', '信息提取', '🔍', stage_num, TOTAL_STAGES)
            yield {'type': 'reasoning_start'}
            yield {'type': 'reasoning', 'content': '🔍 正在提取简历中的文字信息...\n🔍 正在提取JD中的文字信息...\n\n'}

            from concurrent.futures import ThreadPoolExecutor
            ocr_results = {'resume_result': None, 'jd_result': None,
                           'resume_error': None, 'jd_error': None}
            ocr_done = threading.Event()

            def run_ocr():
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_resume = executor.submit(
                        run_tool, extract_text_from_image, {'image_path': resume_path})
                    future_jd = executor.submit(
                        run_tool, extract_text_from_image, {'image_path': jd_path})
                    status_r, result_r = future_resume.result()
                    status_j, result_j = future_jd.result()
                if status_r == 'error':
                    ocr_results['resume_error'] = result_r
                else:
                    ocr_results['resume_result'] = result_r
                if status_j == 'error':
                    ocr_results['jd_error'] = result_j
                else:
                    ocr_results['jd_result'] = result_j
                ocr_done.set()

            threading.Thread(target=run_ocr, daemon=True).start()

            while not ocr_done.is_set():
                yield {'type': 'keepalive'}
                time.sleep(0.5)

            if ocr_results['resume_error']:
                yield {'type': 'error', 'content': f'简历识别失败: {ocr_results["resume_error"]}'}
                return
            if ocr_results['jd_error']:
                yield {'type': 'error', 'content': f'JD识别失败: {ocr_results["jd_error"]}'}
                return

            resume_result = ocr_results['resume_result']
            jd_result = ocr_results['jd_result']

            yield {'type': 'reasoning', 'content': '✅ 文字信息提取完成\n\n'}
        else:
            yield {'type': 'reasoning_start'}
            resume_result = user_input
            jd_result = user_input

        stage_num += 1
        yield from yield_stage('resume_analysis', '简历解析与概括', '📄', stage_num, TOTAL_STAGES)
        if has_image_input:
            yield {'type': 'reasoning', 'content': '\n\n---\n\n'}

        resume_prompt = f"""根据以下简历提取关键信息，进行有详有略的概括总结。突出重点经历与核心技能，适度取舍细节，避免逐条罗列：

{resume_result[:1500]}"""

        yield from stream_llm(resume_prompt)

        stage_num += 1
        yield from yield_stage('jd_analysis', 'JD解析与概括', '📋', stage_num, TOTAL_STAGES)
        yield {'type': 'reasoning', 'content': '\n\n---\n\n'}

        jd_prompt = f"""根据以下职位描述，概括核心要求，有详有略：

{jd_result[:1500]}"""

        yield from stream_llm(jd_prompt)

        stage_num += 1
        yield from yield_stage('match', '匹配分析与反思', '🎯', stage_num, TOTAL_STAGES)
        yield {'type': 'reasoning', 'content': '\n\n---\n\n正在进行技能匹配分析...\n\n'}

        status, match_result_val = run_tool(match_resume_to_jd, {
            'resume_text': resume_result,
            'jd_text': jd_result
        })
        if status == 'error':
            yield {'type': 'error', 'content': f'匹配分析失败: {match_result_val}'}
            return

        match_result_str = str(match_result_val)

        match_prompt = f"""基于以下数据，先分析匹配度（得分、匹配/缺失的技能），再评估匹配结果的合理性。有详有略，基于数据：

- 匹配结果：{match_result_str[:800]}"""

        yield from stream_llm(match_prompt)

        stage_num += 1
        yield from yield_stage('learning', '提升建议', '📚', stage_num, TOTAL_STAGES)
        yield {'type': 'reasoning', 'content': '\n\n---\n\n'}

        match_data = None
        try:
            json_match = re.search(r'\{[\s\S]*\}', match_result_str)
            if json_match:
                match_data = json.loads(json_match.group())
        except:
            pass

        missing_skills = match_data.get('required_skills_missing', []) if match_data else []
        recommendations = match_data.get('recommendations', []) if match_data else []

        learning_prompt = f"""基于匹配结果为求职者提供提升建议，包括学习方向和行动计划。有详有略：

- 缺失技能：{missing_skills if missing_skills else '无'}
- 已有建议：{recommendations if recommendations else '无'}"""

        yield from stream_llm(learning_prompt)

        stage_num += 1
        yield from yield_stage('summary', '生成报告', '📊', stage_num, TOTAL_STAGES)
        yield {'type': 'reasoning_end'}
        yield {'type': 'final_result', 'content': match_result_str}

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield {'type': 'error', 'content': str(e)}
