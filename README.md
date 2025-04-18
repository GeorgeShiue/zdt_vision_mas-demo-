# ZDT_VISION_MAS
**請在根目錄執行以下所有指令**

## 環境設定
1. **創建環境**
    ```bash
    conda env create -f environment.yml # 若成功便可省略安裝套件內的 pip install 步驟
    conda create --name zdt_vision_mas python=3.12.8 # 若上一行指令失敗便採用此創建方法
    conda activate zdt_vision_mas
    ```

2. **安裝套件**
    * pip 安裝
        ```bash
        pip install -r requirements.txt
        ```
    * 將 extend_autogen_package 內的 extend_autogen 資料夾和 setup.py 放到根目錄後執行以下指令
        ```bash
        python setup.py install
        ```

3. **下載語言模型**
    * 下載 SLM 和 VLM
        ```bash
        ollama pull llama3.2:3b
        ollama pull llava:7b
        ```

## 使用步驟

1. **啟動 App**

    **注意：請在根目錄執行以下指令**
    ```bash
    fastapi dev mas/app.py
    ```

2. **新增連接埠**

    在VScode終端機旁邊的連接埠新增預設 port: 8000 和 8080 供前後端運行

3. **開啟網頁前端**
   
   預設網址為: http://127.0.0.1:8000