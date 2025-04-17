import os
import shutil
from dotenv import load_dotenv

import autogen
from autogen.io.websockets import IOWebsockets
from autogen.agentchat.contrib.capabilities.agent_capability import AgentCapability

from extend_autogen import (
    TeamChat,
    SmartGraphBuilder,
    StaffConversableAgent,
    TeamChatManager,
    create_staff_agent_class,
    StaffUserProxyAgent,
)

from preprocess_tools import Tools
from preprocess import agent_relations, agent_dict

# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")
# os.environ["OPENAI_API_KEY"] = api_key

uploaded_image_path = None

def on_connect(iostream: IOWebsockets) -> None:
    print(f" - on_connect(): Connected to client using IOWebsockets {iostream}", flush=True)

    print(" - on_connect(): Receiving message from client.", flush=True)

    # 1. Receive Initial Message
    initial_msg = iostream.input()
    print("message: ", initial_msg, flush=True)
    print("uploaded_image_path: ", uploaded_image_path, flush=True)

    try:
        # 動態宣告Tools
        tools = Tools()
        register_tool = tools.register_tool

        # 動態綁定Tools
        # agent_dict["Preprocessor"].SelfToolingAbility.add_tool(tool_list["image_denoise"])
        register_tool(
            agent_dict,
            ["image_denoise", "image_roi"],
            "Preprocessor",
            "Preprocessing Executor"
        )

        agents = list(agent_dict.values()) # 動態宣告Agents

        # TODO 可能要自訂GroupChat
        team_chat = TeamChat(
            "Preprocess Team",
            agents=agents,
            messages=[],
            allow_repeat_speaker=True,
            agent_relations=agent_relations,
            start_agent=agent_dict["Detector"],
            max_round=50
        )

        chat_manager = TeamChatManager(team_chat)

        print(
            f" - on_connect(): Initiating chat using message '{initial_msg}'",
            flush=True,
        )

        agent_dict["User"].initiate_chat(
            chat_manager, 
            message = initial_msg + f"\n<img {uploaded_image_path}>",
        )
    except Exception as e:
        print(f" - on_connect(): Exception: {e}", flush=True)
        raise e
    


# ----- WebSocket Server ----

from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

PORT = 8000

def load_html_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# on_connect is seperatee executed in a different thread
@asynccontextmanager
async def run_websocket_server(app): 
    with IOWebsockets.run_server_in_thread(on_connect=on_connect, port=8080) as uri:
        print(f"Websocket server started at {uri}.", flush=True)

        yield

app = FastAPI(lifespan=run_websocket_server)

# 設定 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 或指定 ["http://localhost:8000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def get():
    html = load_html_file("mas/frontend/index.html")
    # html = load_html_file("mas/frontend/test_index.html")
    return HTMLResponse(html)

UPLOADED_IMAGES_FOLDER = "mas/frontend/uploaded_images"
PREPROCESSED_IMAGES_FOLDER = "mas/frontend/preprocessed_images"
CACHE_FOLDER = ".cache"

@app.get("/get-agents")
async def get_agents():
    agents_in_relations = []
    for relation in agent_relations:
        for agent in relation:
            # 排除 Executor or Responder
            if "Executor" in agent.name or "Responder" in agent.name:
                continue
            # 確保 agent 存在於 agent_dict 中，且不重複添加
            if agent in agent_dict.values() and agent.name not in agents_in_relations:
                agents_in_relations.append(agent.name)

    print(f"Agents: {agents_in_relations}", flush=True)

    return {"agents": agents_in_relations}

@app.post("/clear-folder")
async def clear_folder():
    if os.path.exists(UPLOADED_IMAGES_FOLDER):
        shutil.rmtree(UPLOADED_IMAGES_FOLDER)
    os.makedirs(UPLOADED_IMAGES_FOLDER, exist_ok=True)
    print(f"Folder '{UPLOADED_IMAGES_FOLDER}' has been cleared.", flush=True)
    
    if os.path.exists(PREPROCESSED_IMAGES_FOLDER):
        shutil.rmtree(PREPROCESSED_IMAGES_FOLDER)
    os.makedirs(PREPROCESSED_IMAGES_FOLDER, exist_ok=True)
    print(f"Folder '{PREPROCESSED_IMAGES_FOLDER}' has been cleared.", flush=True)

    if os.path.exists(CACHE_FOLDER):
        shutil.rmtree(CACHE_FOLDER)
        print(f"Folder '{CACHE_FOLDER}' has been cleared.", flush=True)

    return {"info": f"All folders have been cleared."}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # 清空目標資料夾
    if os.path.exists(UPLOADED_IMAGES_FOLDER):
        shutil.rmtree(UPLOADED_IMAGES_FOLDER)
    os.makedirs(UPLOADED_IMAGES_FOLDER, exist_ok=True)  # 重新建立空資料夾

    # 儲存上傳的圖片
    file_path = os.path.join(UPLOADED_IMAGES_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())  # 將圖片內容寫入檔案

    global uploaded_image_path
    uploaded_image_path = file_path

    print(f"File '{file.filename}' saved at '{file_path}'", flush=True)

    return {"info": f"File '{file.filename}' saved at '{file_path}'"}

@app.get("/get-image")
async def get_image():
    folder_path = PREPROCESSED_IMAGES_FOLDER

    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail="資料夾不存在")

    files = os.listdir(folder_path)
    if not files:
        raise HTTPException(status_code=404, detail="資料夾中沒有圖片檔案")

    file_path = os.path.join(folder_path, files[0])
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="指定的檔案不存在或不是有效檔案")
    
    print(f"File '{file_path}' is being sent.", flush=True)

    return FileResponse(file_path)