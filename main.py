import sys
from agent import create_jobfit_agent
from langchain.messages import HumanMessage


def main():
    print("🤖 JobFit Agent 启动！")
    print("请输入你的需求。例如：")
    print("  - 分析我的简历和JD的匹配度。简历路径: /path/to/resume.png, JD路径: /path/to/jd.png")
    print("  - 这些缺失的Python技能怎么学？")
    print("输入 'quit' 退出。\n")

    agent = create_jobfit_agent()
    # 使用一个固定的 thread_id 来保持同一会话的记忆
    config = {"configurable": {"thread_id": "jobfit_user_session"}}

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            break
        if not user_input:
            continue

        msg = HumanMessage(content=user_input)
        print("正在调用 Agent，请稍候...")  # 新增
        try:
            response = agent.invoke({"messages": [msg]}, config)
            print("调用完成。")  # 新增
            last_msg = response["messages"][-1]
            print(f"Agent: {last_msg.content}")
        except Exception as e:
            print(f"❌ 出错: {e}")
            print("请检查网络、API Key 或图片路径是否正确。")


if __name__ == "__main__":
    main()