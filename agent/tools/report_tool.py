"""
报告生成工具 - 生成匹配分析报告
特性：
1. 生成 Markdown 格式报告
2. 包含匹配分数、技能匹配情况、学习建议
3. 自动保存到报告目录
"""

import json
import os
from datetime import datetime
from langchain_core.tools import tool
from config.settings import settings

@tool
def generate_report(match_json_str: str, additional_advice: str = "") -> str:
    """
    根据匹配结果生成一份 Markdown 格式的报告，并保存到 reports/ 目录。
    
    Args:
        match_json_str: 匹配结果的 JSON 字符串
        additional_advice: 额外建议，如搜索摘要（可选）
        
    Returns:
        报告的文件路径。
    """
    # 解析匹配结果
    try:
        data = json.loads(match_json_str)
    except json.JSONDecodeError:
        data = {}
    
    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(settings.REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(settings.REPORTS_DIR, f"jobfit_report_{timestamp}.md")

    # 生成报告内容
    recommendations = data.get('recommendations', [])
    if isinstance(recommendations, list):
        recommendations = [rec.strip() for rec in recommendations if rec.strip()]
        # 清理建议中可能存在的 Markdown 标题符号
        recommendations = [rec.lstrip('#').strip() for rec in recommendations]
    else:
        recommendations = []
    
    # 清理补充建议中的 Markdown 标题符号
    additional_advice_clean = additional_advice.strip() if additional_advice else '无'
    if additional_advice_clean and additional_advice_clean != '无':
        additional_advice_clean = additional_advice_clean.lstrip('#').strip()
    
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
{chr(10).join(f'- {rec}' for rec in recommendations) if recommendations else '无'}

## 补充建议
{additional_advice_clean}
"""
    
    # 保存报告
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    return f"报告已生成：{report_path}"
