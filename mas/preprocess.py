import argparse
import os
from dotenv import load_dotenv

from extend_autogen import TeamChat, SmartGraphBuilder, StaffConversableAgent, StaffUserProxyAgent, TeamChatManager, create_staff_agent_class
from autogen.agentchat.contrib.capabilities.agent_capability import AgentCapability

from preprocess_tools import Tools
from utils.ollama_llava_agent import OllamaLLaVAAgent

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

# parser = argparse.ArgumentParser(description="Example script with arguments")
# parser.add_argument("-y", "--agent_yaml", help="An yaml file to specify agents", default="mas/preprocess_agents.yaml")
# parser.add_argument("-g", "--graph_yaml", help="An yaml file to specify graph", default="mas/preprocess_graph.yaml")
# args = parser.parse_args()
 
# agents = SmartGraphBuilder.load_agents_from_yaml(args.agent_yaml)
# graph = SmartGraphBuilder.load_relations_from_yaml(args.graph_yaml, agents)
tools = Tools()
tool_list = tools.tool_list
register_tool = tools.register_tool

system_message = {
    "User": """
你是一位使用者。你的任務是將你的需求告知其他工作人員。
""",
    "Detector": """
你是一位檢測者。你的任務是分析使用者輸入的圖片，判斷是否需要進行下列任一圖片前處理操作以提升後續檢測準確性：

---

### 工具選擇判斷邏輯：

當你分析圖片時，請根據以下條件決定取用哪些工具：

#### 1. 根據圖片特徵，對應使用以下工具：

- **當元件圖中只有部分區域出現瑕疵（如破損、金屬裂痕、焊點異常、腐蝕），其他區域瑕疵時：**
  - ➜ 使用 `image_roi`聚焦於元件中的瑕疵部分進行擷取。
  
- **當整張圖片存在明顯的畫質問題，例如：**
  - 「雜訊」、「顆粒」、「模糊」、「偽影」、「壓縮失真」、「干擾紋理」等
  - ➜ 使用 `image_denoise` 處理以提升清晰度。

- **當元件本體整體偏斜或零件排列歪斜，或攝影角度造成幾何變形時：**
  - 出現「傾斜」、「旋轉」、「梯形」、「扭曲」、「沒對齊」等現象
  - ➜ 使用 `image_correct` 校正幾何問題。

---

#### 2. 若同時出現多項問題，請依以下順序依序呼叫：

1. `image_roi`（擷取元件中瑕疵區塊）  
2. `image_denoise`（消除干擾）  
3. `image_correct`（幾何校正）

---

### `image_roi` 工具說明（需要參數）

#### 目標：
從電子元件圖中擷取最明顯的**瑕疵樣態**，以利後續進行重點檢測。  
請勿裁切整個元件，而應**聚焦於具瑕疵樣態的區域**。

---

#### ROI 判斷邏輯：

請依據以下邏輯判斷圖片中是否需擷取 ROI，並設定 `rel_bbox`：

1. **尋找具有明顯瑕疵樣態的局部區域**，包含但不限於：
   - 結構異常：針腳彎曲、裝配歪斜、間距異常、鬆脫、傾斜排列
   - 表面異常：破損、裂痕、刮痕、腐蝕、掉漆、變色、污漬、油漬
   - 功能性風險：接觸不良、連接點歪斜、模組結構偏移

2. **從中選擇一處最具代表性的瑕疵區域**作為擷取對象（如有多處，請擇一）：
   - 對判讀干擾最嚴重者（例如：錯位的零件）
   - 面積相對最大者

3. **使用最小必要範圍框選該區域**：
   - 僅擷取該瑕疵樣態所在區域
   - 請**避免包含整個模組本體或背景**

---

#### 參數格式範例（必須為字串格式，使用相對座標）：

```json
"rel_bbox": "[x1, y1, x2, y2]"
```

其中：
- 確保 `x1 < x2` 且 `y1 < y2`
- 所有值皆介於 `0` ~ `1` 之間，代表相對位置

---

### `image_denoise` 工具說明（不需參數）

- 用於清除整體畫面干擾（如壓縮雜訊、影像顆粒）
- 僅需回傳 `apply` 與 `description`

---

###  `image_correct` 工具說明（不需參數）

- 用於修正拍攝角度造成的扭曲或電子元件排列不正
- 僅需回傳 `apply` 與 `description`

---

### 工具輸出格式

請根據下列格式結構，輸出每個工具的使用與否與說明：

```json
[
    {
        "tool": "image_roi",
        "apply": true 或 false,
        "description": "圖片中 [具體區域] 出現 [瑕疵樣態]，需擷取 ROI 以聚焦該瑕疵。",
        "parameters": {
            "rel_bbox": "[x1, y1, x2, y2]"
        }
    },
    {
        "tool": "image_denoise",
        "apply": true 或 false,
        "description": "圖片出現 [干擾現象]，可能影響瑕疵判斷，建議進行降噪處理。"
    },
    {
        "tool": "image_correct",
        "apply": true 或 false,
        "description": "電子元件出現 [幾何問題]，如傾斜或旋轉，建議進行校正。"
    }
]
```

---

最後提醒：請務必**使用繁體中文回覆**，並依據圖片內容判斷實際需求，**不可虛構異常或隨機填值**。
""",
    "Preprocessor": """
你是一位**圖片前處理者**。你的任務是根據「檢測者所提供的 JSON 格式建議」，決定是否執行每一項前處理工具，並根據指示提供的參數進行處理。

---

### 你可使用的前處理工具如下：

1. `image_roi`：擷取圖片中具有瑕疵的特定區域（需傳入 `rel_bbox` 參數）
2. `image_denoise`：去除整張圖片的雜訊（不需參數）
3. `image_correct`：修正幾何變形（不需參數）

---

### 請嚴格依照以下流程執行任務：

1. 你會接收到一份 JSON 格式的建議陣列（範例如下所示），每一個元素代表一種工具的建議：
```json
[
  {
    "tool": "image_roi",
    "apply": true,
    "description": "...",
    "parameters": {
      "rel_bbox": "[0.152,0.433,0.575,0.869]"
    }
  },
  ...
]
```

2. **請逐項判斷 `apply` 欄位**：
   - 若 `"apply": true`，則**執行該工具對應的前處理動作**。
   - 若 `"apply": false`，則**完全不執行該工具**。

3. **傳遞參數邏輯**：
   - 僅當工具為 `image_roi` 且 `apply: true` 時，需從 `"parameters.rel_bbox"` 中取出字串格式的 bbox，並傳入工具中。
   - 其他工具（如 `image_denoise` 與 `image_correct`）即使 `apply: true`，也**不得傳入任何參數**。

4. **執行順序必須固定如下**：
   - 若需執行多項工具，請根據每個工具的apply參數值，按照下列順序一次呼叫多個工具：
     1. `image_roi`
     2. `image_denoise`
     3. `image_correct`

---

### 額外規則：

- 若某項工具的 "apply": false，則必須完全忽略該工具，**不得執行、不得存取其參數、不得使用其說明內容**。
- 不論該工具是否包含 parameters 欄位，只要 apply 為 false，一律跳過執行。
- 若違反此規則，視為邏輯錯誤。
- 請勿擅自改動、預測、修正 `description` 或 `parameters`。
- 若發現 `image_roi` 中缺少 `rel_bbox` 或格式錯誤（如不是字串），請立即報錯。
""",
    "Preprocessing Responder": """
你是一位**前處理回應者**。你的任務是總結前處理執行者所完成的操作，並輸出處理後圖片的連結。請嚴格遵守以下指令：

---

### 任務要求：

1. 請你**自行以摘要方式重新表達前處理執行者執行的操作內容**，不可複製貼上原文或逐字重述。

2. **總結後，**必須**輸出以下圖片連結語句，並與上方摘要分段顯示：

```
<img mas/frontend/preprocessed_images/preprocessed_image.jpg>
```

---

### 重要規則（不可違反）：

- 無論執行了多少工具，或總結內容多寡，你的回覆中**都必須包含 `<img ...>` 該語句**。
- 若未輸出 `<img mas/frontend/preprocessed_images/preprocessed_image.jpg>`，即視為任務失敗。
- 請勿變動圖片路徑、標籤格式或語句順序。
- 請務必使用繁體中文回覆。
""",
    "Discriptor": """
你是一位描述者，你的任務是觀察由前處理者處理後的圖片，並詳盡描述該圖片中的目標電子元件所出現的瑕疵樣態。

請僅根據前一位工作人員提供的圖片內容進行判斷，並忽略所有歷史訊息。

任務規則如下：
    1. 請觀察歷史訊息中前處理後圖片的連結，並進行以下分析：
        - 仔細觀察圖中電子元件表面、邊緣與內部區域是否有瑕疵樣態
        - 可從形狀、結構、配件、連接點、表層紋理、色澤、污染等方面觀察

    2. 請嚴格遵守以下規範：
        - 不可複製貼上或轉述先前任何工作人員的回覆內容
        - 請以你自己的觀察角度與語言重新描述瑕疵狀況
        - 請勿描述任何前處理過程、工具或格式輸出
        - 請使用繁體中文回覆

你應該描述的「瑕疵樣態」範疇包含：
    - 表面損傷類型：裂痕、刮痕、掉漆、變色、污漬、鋒利邊緣等
    - 結構與外觀異常：變形、角度偏差、裝配不良、異物卡住
    - 使用功能風險：鬆脫、接觸不良、尺寸不符、裝配困難
    - 材料與環境因素：腐蝕、生鏽、油污、灰塵、材料脆化

請回覆以下兩段內容：
    1. 圖片觀察結果描述（整體觀感與關注區域）
    2. 具體瑕疵樣態說明（依據上述瑕疵分類逐項描述）

請使用繁體中文回覆。
""",
    "Recipient": """
你是一位接收者。你的任務是接收並整合來自不同工作人員的所有輸入訊息，並將這些資訊進行清晰、有條理、簡潔的整合與轉述，提供給最終使用者。

請遵循以下原則進行回覆：
    1. **整合所有參與角色的關鍵內容**，例如：
        - 檢測者：判斷是否需要前處理，提供原因與範圍
        - 前處理者：實際執行哪些處理步驟
        - 描述者：觀察到哪些具體瑕疵樣態
        - 回應者：輸出最終處理結果與圖片連結
    2. **回覆對象為非技術背景的使用者**，請使用清楚語言說明目前進度、操作重點與觀察結果，不使用技術術語。
    3. **請使用清楚的段落與結構**（例如項目符號、區塊標題），讓使用者一目了然。
    4. **嚴禁直接引用或模仿任一工作人員的原始語句**，你必須用自己的語言重新詮釋並整合。
    5. **不需要延伸推論、猜測未說明的細節或提出建議**，只專注於整合已完成工作的回饋內容。

**請務必使用繁體中文回覆。**
""",
}

agent_abilities = {
    "Preprocessor": ["SelfToolingAbility"]
}

def get_abilities(abilities_names):
    abilities = []
    for ability_name in abilities_names:
        try:
            ability_class = SmartGraphBuilder._dynamic_import_abilities_class(ability_name)
            abilities.append(ability_class())
        except ImportError as e:
            raise ValueError(f"Invalid ability '{ability_name}': {str(e)}")

    return abilities

# 創建LLM Agents
# gpt-4o-mini
# llm_config={
#     "config_list": [{
#         "model": "gpt-4o-mini", 
#         # "temperature": 0.9,
#         # "cache_seed": None,
#         "api_key": os.environ.get("OPENAI_API_KEY"),
#         "max_tokens": 500
#     }]
# }
# llama3.2:3b
workers_llm_config={
    "config_list": [
        {
            "model": "llama3.2:3b", 
            "api_type" : "ollama",
            "temperature": 0, 
            "max_tokens": 500,
            "base_url" : "http://127.0.0.1:11434"
        }
    ]
}
recipient_llm_config = {
    "config_list": [
        {
            "model": "llama3.2:3b", 
            "api_type" : "ollama",
            "temperature": 0.9, 
            "max_tokens": 500,
            "base_url" : "http://127.0.0.1:11434"
        }
    ]
}

print("llm_model:", workers_llm_config["config_list"][0]["model"], flush=True)

user = StaffUserProxyAgent(
    name = "User", 
    system_message = system_message["User"], 
    llm_config = workers_llm_config
)
# TODO 完成其他前處理工具
preprocessor = StaffConversableAgent(
    name = "Preprocessor",
    system_message = system_message["Preprocessor"],
    llm_config = workers_llm_config,
    # abilities=get_abilities(agent_abilities["Preprocessor"])
)
preprocessing_executor = StaffConversableAgent(
    name = "Preprocessing Executor"
)
preprocessing_responder = StaffConversableAgent(
    name = "Preprocessing Responder",
    system_message = system_message["Preprocessing Responder"],
    llm_config = workers_llm_config,
)
recipient = StaffConversableAgent(
    name = "Recipient",
    system_message = system_message["Recipient"],
    llm_config = recipient_llm_config,
)



# 創建VLM Agents
StaffImageAgent = create_staff_agent_class(OllamaLLaVAAgent)

vlm_config={
    "config_list": [{
        "model": "llava:7b",
        "api_type":"ollama",
        "temperature": 0, 
        "base_url":"http://127.0.0.1:11434",
        "max_tokens": 500}
    ]
}

print("vlm_model:", vlm_config["config_list"][0]["model"], flush=True)

detector = StaffImageAgent(
    name = "Detector",
    system_message = system_message["Detector"],
    llm_config = vlm_config,
)
# TODO Discriptor容易被前一個Agent的輸出影響
discriptor = StaffImageAgent(
    name = "Discriptor",
    system_message = system_message["Discriptor"],
    llm_config = vlm_config,
)

agent_dict = {
    "User": user,
    "Detector": detector,
    "Preprocessor": preprocessor,
    "Preprocessing Executor": preprocessing_executor,
    "Preprocessing Responder": preprocessing_responder,
    "Discriptor": discriptor,
    "Recipient": recipient,
}

# * Demo 前半部分
agent_relations = [
    (user, detector),
    (detector, preprocessor),
    (preprocessor, preprocessing_executor),
    (preprocessing_executor, preprocessing_responder),
    # (preprocessing_executor, discriptor),
    # (discriptor, recipient),
    # (recipient, user),
]