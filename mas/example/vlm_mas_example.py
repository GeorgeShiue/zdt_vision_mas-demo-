from autogen.agentchat.contrib.llava_agent import LLaVAAgent, llava_call
import autogen
from ollama_llava_agent import OllamaLLaVAAgent

# llm_config = autogen.LLMConfig(
#     model="llava:7b",
#     api_type="ollama",
#     stream=False,
#     base_url="http://127.0.0.1:11434",
# )
llm_config={"config_list": [{"model": "llava:7b",  "api_type":"ollama", "temperature": 0.9,  "base_url":"http://127.0.0.1:11434","max_tokens": 500}]}


image_agent = OllamaLLaVAAgent(
    name="image-explainer",
    system_message="""
你是一位檢測者。你的任務是評估輸入的圖片，判斷是否需要以下任一前處理操作：
    - 降噪：用於移除雜訊或壓縮產生的干擾。
    - 校正：用於修正傾斜、旋轉或幾何變形。
    - ROI 擷取：用於擷取有意義的區域，去除無關背景。

當你分析圖片時，請依照以下判斷邏輯：
    - 若圖片中存在明顯雜訊、顆粒感、壓縮偽影，或整體模糊，請建議執行 降噪。
    - 若圖片看起來傾斜、歪斜、變形（如梯形或拉伸），或未對齊，請建議執行 校正。
    - 若圖片中大部分為背景，且主要內容僅佔很小一部分（例如：發票、表單、繁雜場景中的物體），請建議執行 ROI 擷取。
    
若圖片同時符合多項狀況，也可以同時建議多種前處理方式。

    你的回覆應包含：
    - 對觀察到問題的簡要描述。
    - 建議的前處理方式。
    
記得跟每個人先打招呼，要說出名稱。
""",
    max_consecutive_auto_reply=10,
    llm_config=llm_config
)

user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="A human admin.",
    code_execution_config={
        "last_n_messages": 3,
        "work_dir": "groupchat",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    human_input_mode="NEVER",  # Try between ALWAYS or NEVER
    max_consecutive_auto_reply=0,
)

# # Ask the question with an image
# user_proxy.initiate_chat(
#     image_agent,
#     message="""
# <img sd3_fintune/input/test_PCB1/000.JPG>
# """,
# )
# * 圖片輸入可以傳入檔案相對路徑