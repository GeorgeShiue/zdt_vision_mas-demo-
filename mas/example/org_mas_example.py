from extend_autogen import TeamChat, StaffConversableAgent, TeamChatManager
import os

llm_config={"config_list": [{"model": "gpt-4o-mini", "temperature": 0.9, "api_key": os.environ.get("OPENAI_API_KEY"),"max_tokens": 500}]}
system_message =[
    "你是專案經理。你的責任是確保整個專案的進度，並協調開發、測試、設計、以及市場策略。你需要掌握項目的整體情況，並確保所有的工作按時完成。記得跟每個人先打招戶，要說出名稱。",
    "你是開發主管。你的責任是管理專案的技術開發工作，確保技術需求能夠順利實現。你需要提供開發進度的更新，並解決技術挑戰以推進專案。記得跟每個人先打招戶，要說出名稱。",
    "你是技術協調者。你的任務是掌握開發和測試的技術狀況，總結進度和遇到的挑戰，並協助各方解決問題，確保專案順利進行。記得跟每個人先打招戶，要說出名稱。",
    "你是測試主管。你的責任是確保所有的產品功能都經過全面測試，並且符合質量標準。你需要提供測試進度，並確保測試計劃順利實施。記得跟每個人先打招戶，要說出名稱。",
    "你負責協調設計與市場策略。你的任務是根據技術進度和測試結果，制定適合的市場推廣策略，並確保設計符合產品需求和市場預期。記得跟每個人先打招戶，要說出名稱。",
    "你是產品經理。你的任務是根據開發、測試和市場反饋，做出關於產品發布的最終決策，確保產品符合技術要求並能夠滿足市場需求。記得跟每個人先打招戶，要說出名稱。"
]
agent_a = StaffConversableAgent(name="Project_Manager", system_message=system_message[0], llm_config=llm_config)
agent_b = StaffConversableAgent(name="Lead_Developer", system_message=system_message[1], llm_config=llm_config)
agent_c = StaffConversableAgent(name="Technical_Coordinator", system_message=system_message[2], llm_config=llm_config)
agent_d = StaffConversableAgent(name="Lead_Tester", system_message=system_message[3], llm_config=llm_config)
agent_e = StaffConversableAgent(name="Marketing_and_Design_Integrator", system_message=system_message[4], llm_config=llm_config)
agent_f = StaffConversableAgent(name="Product_Manager", system_message=system_message[5], llm_config=llm_config)

team_chat = TeamChat(
    "team A",
    agents=[agent_a, agent_b, agent_c, agent_d, agent_e, agent_f],
    messages=[],
    allow_repeat_speaker=True,
    agent_relations=[
        (agent_a, agent_b),
        (agent_b, agent_c),
        (agent_c, agent_d),
        (agent_d, agent_e),
        (agent_e, agent_f)
    ],
    start_agent=agent_a
)

chat_manager = TeamChatManager(team_chat)
groupchat_result = agent_a.initiate_chat(
    chat_manager, message="""
大家好，我們接到了一個新的專案，目的是開發一個能夠改善用戶日常生活的健康管理應用程式。我需要大家的專業建議來完成這個提案，確保我們能夠按時推出高品質的產品。
我們需要在一週內完成這個提案，所以請大家分享目前的想法和計劃，並確保我們的工作進度能夠協調一致。
"""
)

with open("./tmp/sequential.txt", "w", encoding="utf-8") as file:
    for message in team_chat.messages:
        file.write(f"{message['name']}: {message['content']}\n\n\n")