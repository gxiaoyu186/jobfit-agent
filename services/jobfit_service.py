"""
JobFit 服务 - 处理简历与JD匹配的核心业务逻辑
特性：
1. 封装匹配分析的业务流程
2. 使用 Agent 进行智能决策
3. 支持流式响应
4. 统一异常处理
5. 报告存储管理（按用户存储，最多10个）
"""

import json
import os
import re
import time
from datetime import datetime
from typing import Generator, Optional, List, Dict

from config.settings import settings
from agent.jobfit_agent import create_jobfit_agent, stream_jobfit_agent
from exceptions.jobfit_exceptions import ValidationException, ModelException

class AnalysisResult:
    """分析结果数据类"""
    
    def __init__(self, match_score: int, required_skills_match: list, 
                 required_skills_missing: list, preferred_skills_match: list,
                 preferred_skills_missing: list, recommendations: list):
        self.match_score = match_score
        self.required_skills_match = required_skills_match
        self.required_skills_missing = required_skills_missing
        self.preferred_skills_match = preferred_skills_match
        self.preferred_skills_missing = preferred_skills_missing
        self.recommendations = recommendations
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            'match_score': self.match_score,
            'required_skills_match': self.required_skills_match,
            'required_skills_missing': self.required_skills_missing,
            'preferred_skills_match': self.preferred_skills_match,
            'preferred_skills_missing': self.preferred_skills_missing,
            'recommendations': self.recommendations
        }

class JobFitService:
    """JobFit 核心服务类"""

    def __init__(self):
        """初始化 Agent 实例（懒加载）"""
        self._agent = None
        self._model = None

    @property
    def agent(self):
        """懒加载 Agent 实例"""
        if self._agent is None:
            self._agent, self._model = create_jobfit_agent()
        return self._agent

    @property
    def model(self):
        """懒加载模型实例"""
        if self._model is None:
            self._agent, self._model = create_jobfit_agent()
        return self._model
    
    def analyze_match(self, resume_path: str = "", jd_path: str = "", 
                      message: str = "分析匹配度", username: str = "anonymous",
                      auto_generate_report: bool = True) -> Generator[dict, None, None]:
        """
        分析简历与 JD 的匹配度（流式响应）
        
        用户可以上传简历和JD图片，同时可以在消息中输入额外的文字说明，
        系统会综合图片识别结果和用户消息进行分析。
        
        Args:
            resume_path: 简历文件路径
            jd_path: JD 文件路径
            message: 用户消息（可包含文字形式的简历和JD内容）
            username: 用户名（用于自动生成报告）
            auto_generate_report: 是否自动生成报告
        
        Yields:
            流式响应数据块
            
        Raises:
            ValidationException: 参数验证失败
            ModelException: 模型调用失败
        """
        try:
            has_image_input = resume_path and jd_path

            if not message and not has_image_input:
                raise ValidationException('请上传简历和JD图片，或在输入框中输入内容')

            if resume_path and not os.path.exists(resume_path):
                raise ValidationException(f'简历文件不存在: {resume_path}')

            if jd_path and not os.path.exists(jd_path):
                raise ValidationException(f'JD文件不存在: {jd_path}')

            if has_image_input:
                user_input = f"""{message}。

简历路径: {resume_path}
JD路径: {jd_path}

请分析以上内容的匹配度。"""
            else:
                user_input = f"""{message}

请分析以上内容的匹配度。"""

            session_id = f"jobfit_session_{int(time.time())}"
            config = {"configurable": {"thread_id": session_id}}

            match_result = {}

            # 直接调用 Agent 的 stream 模式
            for chunk in stream_jobfit_agent(
                self.agent, self.model, resume_path, jd_path,
                user_input, config, has_image_input
            ):
                chunk_type = chunk.get('type')

                if chunk_type == 'final_result':
                    final_content = chunk.get('content', '')
                    if final_content:
                        try:
                            json_match = re.search(r'\{[\s\S]*\}', final_content)
                            if json_match:
                                match_result = json.loads(json_match.group())
                        except Exception:
                            pass
                    continue

                if chunk_type == 'error':
                    raise ModelException(f'分析失败: {chunk.get("content")}')

                yield chunk

            # 自动生成报告
            if auto_generate_report and match_result and isinstance(match_result, dict) and 'match_score' in match_result:
                try:
                    self.generate_report(match_result, username)
                    yield {'type': 'report_generated', 'content': '报告已自动生成'}
                except Exception:
                    pass

            # 生成基本总结并输出
            if match_result and isinstance(match_result, dict) and 'match_score' in match_result:
                match_score = match_result.get('match_score', 0)
                
                if match_score >= 80:
                    match_status = "高度匹配"
                elif match_score >= 60:
                    match_status = "中度匹配"
                else:
                    match_status = "低度匹配"
                
                required_match = match_result.get('required_skills_match', [])
                required_missing = match_result.get('required_skills_missing', [])
                preferred_match = match_result.get('preferred_skills_match', [])
                preferred_missing = match_result.get('preferred_skills_missing', [])
                recommendations = match_result.get('recommendations', [])
                
                summary_lines = []
                summary_lines.append(f"## 🎯 匹配分析完成")
                summary_lines.append(f"")
                summary_lines.append(f"**综合匹配分数：** {match_score}分 ({match_status})")
                summary_lines.append(f"")
                
                if required_match:
                    summary_lines.append(f"**✅ 必需技能匹配（{len(required_match)}项）：**")
                    for skill in required_match:
                        summary_lines.append(f"  - {skill}")
                    summary_lines.append(f"")
                
                if required_missing:
                    summary_lines.append(f"**❌ 必需技能缺失（{len(required_missing)}项）：**")
                    for skill in required_missing:
                        summary_lines.append(f"  - {skill}")
                    summary_lines.append(f"")
                
                if preferred_match:
                    summary_lines.append(f"**✨ 加分技能匹配（{len(preferred_match)}项）：**")
                    for skill in preferred_match:
                        summary_lines.append(f"  - {skill}")
                    summary_lines.append(f"")
                
                if preferred_missing:
                    summary_lines.append(f"**💪 加分技能待提升（{len(preferred_missing)}项）：**")
                    for skill in preferred_missing:
                        summary_lines.append(f"  - {skill}")
                    summary_lines.append(f"")
                
                if recommendations:
                    summary_lines.append(f"**📚 提升建议：**")
                    for i, rec in enumerate(recommendations, 1):
                        summary_lines.append(f"  {i}. {rec}")
                    summary_lines.append(f"")
                
                summary_lines.append(f"详细分析报告已自动生成，可在左侧「📊 生成报告」按钮中查看完整内容。")
                
                summary_content = '\n'.join(summary_lines)
                yield {'type': 'agent', 'content': summary_content}

            yield {'type': 'done', 'content': ''}
            
        except Exception as e:
            raise ModelException(f'分析失败: {str(e)}')
    
    def generate_report(self, match_result: dict, username: str = "anonymous", additional_advice: str = "") -> str:
        """
        生成匹配分析报告并按用户存储
        
        Args:
            match_result: 匹配结果字典
            username: 用户名（用于按用户存储）
            additional_advice: 额外建议
            
        Returns:
            报告文件路径
        """
        user_report_dir = os.path.join(settings.REPORTS_DIR, username)
        os.makedirs(user_report_dir, exist_ok=True)
        self._cleanup_old_reports(user_report_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"jobfit_report_{timestamp}.md"
        report_path = os.path.join(user_report_dir, report_filename)
        
        recommendations = match_result.get('recommendations', [])
        if isinstance(recommendations, list):
            recommendations = [rec.strip() for rec in recommendations if rec.strip()]
            recommendations = [rec.lstrip('#').strip() for rec in recommendations]
        else:
            recommendations = []
        
        additional_advice_clean = additional_advice.strip() if additional_advice else ''
        if additional_advice_clean:
            additional_advice_clean = additional_advice_clean.lstrip('#').strip()
        
        match_score = match_result.get('match_score', 0)
        
        if match_score >= 80:
            match_status = "🟢 高度匹配"
            match_level = "高"
            match_color = "🟢"
            score_bar = "██████████"
            score_empty = ""
        elif match_score >= 60:
            match_status = "🟡 中度匹配"
            match_level = "中"
            match_color = "🟡"
            filled = match_score // 10
            score_bar = "█" * filled
            score_empty = "░" * (10 - filled)
        else:
            match_status = "🔴 低度匹配"
            match_level = "低"
            match_color = "🔴"
            filled = max(1, match_score // 10)
            score_bar = "█" * filled
            score_empty = "░" * (10 - filled)
        
        required_match = match_result.get('required_skills_match', [])
        required_missing = match_result.get('required_skills_missing', [])
        preferred_match = match_result.get('preferred_skills_match', [])
        preferred_missing = match_result.get('preferred_skills_missing', [])
        
        req_total = len(required_match) + len(required_missing)
        req_rate = f"{len(required_match)}/{req_total} ({round(len(required_match)/req_total*100) if req_total > 0 else 0}%)"
        pref_total = len(preferred_match) + len(preferred_missing)
        pref_rate = f"{len(preferred_match)}/{pref_total} ({round(len(preferred_match)/pref_total*100) if pref_total > 0 else 0}%)"
        
        required_match_rows = '\n'.join(f'| {skill} | ✅ 已具备 |' for skill in required_match) if required_match else '| - | 暂无匹配项 |'
        required_missing_rows = '\n'.join(f'| **{skill}** | ❌ 需补充 | ⭐⭐⭐ 优先学习 |' for skill in required_missing) if required_missing else '| - | 暂无缺失项 | - |'
        preferred_match_rows = '\n'.join(f'| {skill} | ✨ 已具备 |' for skill in preferred_match) if preferred_match else '| - | 暂无匹配项 |'
        preferred_missing_rows = '\n'.join(f'| {skill} | 💪 建议学习 | ⭐ 进阶提升 |' for skill in preferred_missing) if preferred_missing else '| - | 暂无待提升项 | - |'
        
        learning_items = '\n'.join(f'| {i+1} | {rec} |' for i, rec in enumerate(recommendations)) if recommendations else '| - | 暂无建议 |'
        
        priority_skills = required_missing[:3] if required_missing else (preferred_missing[:3] if preferred_missing else [])
        priority_items = '\n'.join(f'| {i+1} | 🔴 高 | {skill} | 建议通过在线课程、实战项目等方式系统学习 |' for i, skill in enumerate(priority_skills)) if priority_skills else '| - | - | 暂无紧急待学技能 | - |'
        
        generated_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        tech_hard_score = min(10, max(1, int((len(required_match) * 10) / max(1, req_total))))
        tech_hard_filled = '█' * tech_hard_score
        tech_hard_empty = '░' * max(0, 10 - tech_hard_score)
        
        exp_score = min(10, max(1, match_score // 10))
        exp_filled = '█' * exp_score
        exp_empty = '░' * max(0, 10 - exp_score)
        
        tech_level = '🟢 扎实' if len(required_match) >= len(required_missing) else ('🟡 需加强' if required_match else '🔴 薄弱')
        match_level_text = '高' if match_score >= 80 else ('中' if match_score >= 60 else '低')
        growth_level = '🟢 良好' if recommendations else '🟡 一般'
        growth_note = f'已有 {len(recommendations)} 条提升建议可供参考' if recommendations else '建议主动学习补充缺失技能'
        compete_level = '强' if match_score >= 80 else ('中等' if match_score >= 60 else '需大幅提升')
        compete_note = '核心竞争力突出，面试成功率较高' if match_score >= 80 else ('具备一定竞争力，补充缺失技能后可大幅提升' if match_score >= 60 else '建议优先补充核心必需技能后再投递')
        
        overall_compete = '强' if match_score >= 80 else ('中等' if match_score >= 60 else '需提升')
        
        short_term_1 = f'- 优先学习 **{required_missing[0]}**，这是岗位最核心的缺失技能' if required_missing else '- 巩固已有技能，准备面试常见问题'
        short_term_2 = f'- 针对 **{required_missing[1]}** 完成一个实战项目' if len(required_missing) > 1 else '- 梳理项目经验，准备 STAR 法则面试回答'
        short_term_3 = f'- 复习 **{required_missing[2]}** 相关知识，准备技术面试' if len(required_missing) > 2 else '- 调研目标公司的技术栈与文化'
        
        mid_term_1 = f'- 系统学习 {"、".join(required_missing)} 等核心技能' if required_missing else '- 拓展加分技能，提升综合竞争力'
        mid_term_2 = f'- 参与开源项目或实战项目，积累 {"、".join(preferred_missing[:2]) if preferred_missing else "相关"} 领域的实际经验' if preferred_missing else '- 关注行业动态，保持技术敏感度'
        
        long_term_header = '| 时间 | 目标 | 具体行动 |' if required_missing or preferred_missing else ''
        long_term_sep = '|:------|:------|:----------|' if required_missing or preferred_missing else ''
        long_term_rows_1 = '\n'.join(f'| 第{i+1}个月 | 掌握 {skill} | 完成相关课程学习与项目实战 |' for i, skill in enumerate(required_missing[:3])) if required_missing else ''
        long_term_rows_2 = '\n'.join(f'| 第{i+1+len(required_missing[:3])}个月 | 提升 {skill} | 深入学习并应用于实际项目 |' for i, skill in enumerate(preferred_missing[:2])) if preferred_missing else ''
        
        summary_header = '### ⚠️ 关键短板' if required_missing else '### ✅ 优势确认'
        summary_shortcomings = '\n'.join(f'- **{skill}**：需作为最优先学习目标，建议制定详细学习计划并定期复盘' for skill in required_missing[:3]) if required_missing else '- 您的必需技能与岗位要求高度匹配，具备较强的核心竞争力'
        
        advantage_skills = required_match[:2] if required_match else (preferred_match[:2] if preferred_match else [])
        advantages = '\n'.join(f'- **{skill}**：已掌握，是您的核心竞争力之一' for skill in advantage_skills) if advantage_skills else '- 建议尽快补充核心技能以建立竞争优势'
        preferred_advantages = '\n'.join(f'- **{skill}**：建议进一步学习以增强面试竞争力' for skill in preferred_missing[:2]) if preferred_missing else ''
        
        deliver_advice = '✅ 当前匹配度较高，建议积极投递并准备面试' if match_score >= 80 else ('⚠️ 匹配度尚有提升空间，建议在补充1-2项核心缺失技能后投递' if match_score >= 60 else '🔴 匹配度较低，建议优先补充核心必需技能，暂缓投递该岗位')
        
        content = f"""# 📊 JobFit 职业匹配分析报告

> 本报告由 AI 智能分析生成，旨在为您提供客观、全面的职业匹配评估与提升建议。

---

## 📅 报告信息

| 项目 | 详情 |
|:------|:------|
| 📆 生成时间 | {generated_time} |
| 👤 分析用户 | **{username}** |
| 🤖 分析引擎 | JobFit Agent v5.0 |

---

## 1. 🎯 匹配结果总览

```
综合匹配分数：{match_score} / 100 分
{match_color} [{score_bar}{score_empty}] {match_status}

等级评定：{match_level}匹配
```

| 评估维度 | 结果 | 说明 |
|:----------|:------|:------|
| 🎯 综合匹配分数 | **{match_score}分** | {match_status} |
| 📊 必需技能达标率 | {req_rate} | 岗位核心要求满足程度 |
| 🌟 加分技能掌握率 | {pref_rate} | 额外竞争力评估 |
| 📈 整体竞争力 | {match_color} {overall_compete} | 综合求职竞争力评级 |

---

## 2. 📋 必需技能匹配详情

> **必需技能** 是岗位的核心硬性要求，直接影响简历筛选通过率。

### ✅ 已匹配的必需技能（{len(required_match)}项）

| 技能名称 | 状态 |
|:----------|:------|
{required_match_rows}

### ❌ 缺失的必需技能（{len(required_missing)}项）

| 技能名称 | 状态 | 学习优先级 |
|:----------|:------|:------------|
{required_missing_rows}

---

## 3. 🌟 加分技能匹配详情

> **加分技能** 虽非硬性要求，但能显著提升您的竞争力与面试通过率。

### ✨ 已掌握的加分技能（{len(preferred_match)}项）

| 技能名称 | 状态 |
|:----------|:------|
{preferred_match_rows}

### 💪 待提升的加分技能（{len(preferred_missing)}项）

| 技能名称 | 状态 | 学习优先级 |
|:----------|:------|:------------|
{preferred_missing_rows}

---

## 4. 📚 学习提升建议

### 🔥 紧急优先学习计划

| 序号 | 优先级 | 技能 | 学习建议 |
|:------|:--------|:------|:----------|
{priority_items}

### 📝 完整建议清单

| 序号 | 建议内容 |
|:------|:----------|
{learning_items}

---

## 5. 📈 综合能力评估

### 五维能力雷达

```
技术硬实力  {tech_hard_filled}{tech_hard_empty}
项目经验    {exp_filled}{exp_empty}
学习能力    {'█' * 7}{'░' * 3}（基于评估推断）
综合匹配    {exp_filled}{exp_empty}
竞争力指数  {exp_filled}{exp_empty}
```

### 综合评估表

| 评估维度 | 评级 | 分析说明 |
|:----------|:------|:----------|
| 🎓 技术硬实力 | {tech_level} | 必需技能匹配 {len(required_match)} 项，缺失 {len(required_missing)} 项 |
| 📋 岗位匹配度 | {match_color} {match_level_text} | 综合得分 {match_score} 分，{match_status} |
| 🚀 成长潜力 | {growth_level} | {growth_note} |
| 🏆 面试竞争力 | {match_color} {compete_level} | {compete_note} |

---

## 6. 💡 求职行动建议

### 🎯 短期行动（1-2周）

{short_term_1}
{short_term_2}
{short_term_3}

### 📅 中期规划（1-3个月）

{mid_term_1}
{mid_term_2}
- 持续投递简历，积累面试经验

### 🏁 长期发展（3-6个月）

{long_term_header}
{long_term_sep}
{long_term_rows_1}
{long_term_rows_2}

---

## 7. 📌 总结

> 本次匹配分析综合得分为 **{match_score}分**，评定为 **{match_status}**。

{summary_header}
{summary_shortcomings}

### 💪 优势领域
{advantages}
{preferred_advantages}

### 🎯 投递建议
{deliver_advice}

---

*📊 报告由 JobFit Agent v5.0 自动生成 | 生成时间：{generated_time}*
"""
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        return report_path
    
    def _cleanup_old_reports(self, user_report_dir: str, max_reports: int = 10):
        """
        清理用户超出数量限制的旧报告
        
        Args:
            user_report_dir: 用户报告目录
            max_reports: 最大报告数量（默认10个）
        """
        if not os.path.exists(user_report_dir):
            return
        
        # 获取所有报告文件并按修改时间排序
        report_files = []
        for filename in os.listdir(user_report_dir):
            if filename.startswith('jobfit_report_') and filename.endswith('.md'):
                filepath = os.path.join(user_report_dir, filename)
                if os.path.isfile(filepath):
                    report_files.append((filepath, os.path.getmtime(filepath)))
        
        # 按修改时间升序排序（最旧的在前）
        report_files.sort(key=lambda x: x[1])
        
        # 删除超出限制的旧报告
        while len(report_files) > max_reports:
            oldest_file = report_files.pop(0)
            os.remove(oldest_file[0])
    
    def get_user_reports(self, username: str) -> List[Dict]:
        """
        获取用户的报告列表
        
        Args:
            username: 用户名
            
        Returns:
            报告列表，包含文件名、路径和创建时间
        """
        user_report_dir = os.path.join(settings.REPORTS_DIR, username)
        reports = []
        
        if os.path.exists(user_report_dir):
            for filename in os.listdir(user_report_dir):
                if filename.startswith('jobfit_report_') and filename.endswith('.md'):
                    filepath = os.path.join(user_report_dir, filename)
                    if os.path.isfile(filepath):
                        # 从文件名提取时间戳
                        timestamp_str = filename.replace('jobfit_report_', '').replace('.md', '')
                        try:
                            created_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        except:
                            created_time = datetime.fromtimestamp(os.path.getctime(filepath))
                        
                        reports.append({
                            'filename': filename,
                            'path': filepath,
                            'created_time': created_time.strftime("%Y-%m-%d %H:%M:%S"),
                            'timestamp': timestamp_str
                        })
        
        # 按创建时间降序排序（最新的在前）
        reports.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return reports
    
    def get_report_content(self, username: str, filename: str) -> Optional[str]:
        """
        获取报告内容
        
        Args:
            username: 用户名
            filename: 报告文件名
            
        Returns:
            报告内容，如果文件不存在返回 None
        """
        user_report_dir = os.path.join(settings.REPORTS_DIR, username)
        filepath = os.path.join(user_report_dir, filename)
        
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        
        return None
