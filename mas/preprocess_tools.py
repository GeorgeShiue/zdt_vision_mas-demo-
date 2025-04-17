import os
import cv2
import numpy as np
import ast
import shutil

import inspect
from autogen import register_function

UPLOADED_IMAGES_FOLDER = "mas/frontend/uploaded_images"
PREPROCESSED_IMAGES_FOLDER = "mas/frontend/preprocessed_images"

class Tools():
    def __init__(self):
        def get_input_image():
            """Get the input image file from the specified directory."""

            # 先找 preprocessed_images 資料夾
            folder_path = PREPROCESSED_IMAGES_FOLDER
            files = os.listdir(folder_path)

            # 過濾出圖片檔案
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not image_files:
                # 如果沒有找到，則找 uploaded_images 資料夾
                folder_path = UPLOADED_IMAGES_FOLDER
                files = os.listdir(folder_path)
                # 過濾出圖片檔案
                image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not image_files:    
                raise FileNotFoundError("No image files found in the specified directory.")

            # 返回第一個圖片檔案的名稱
            return folder_path, image_files[0]
        
        def clear_folder(folder_path: str):
            """Clear the specified folder by removing all files and subdirectories."""
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
            os.makedirs(folder_path, exist_ok=True)
            print(f"Folder '{folder_path}' has been cleared.", flush=True)

        # TODO 之後可以把folder_path改成參數
        def save_preprocessed_image(image: np.ndarray, name: str) -> str:
            """Save the preprocessed image to the specified preprocessed directory."""

            # TODO 保持輸出資料夾內圖片的唯一性
            folder_path = PREPROCESSED_IMAGES_FOLDER
            clear_folder(folder_path)  # 清空資料夾

            path = os.path.join(folder_path, name)
            cv2.imwrite(path, image)

            return path


        def image_denoise() -> str:
            """Denoise the input image using filters."""

            folder_path, input_image_name = get_input_image()
            input_image_path = os.path.join(folder_path, input_image_name)
            img = cv2.imread(input_image_path)
            # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # 方法一：Gaussian Blur（高斯模糊）
            # gaussian = cv2.GaussianBlur(img, (5, 5), 0)

            # # 方法二：Median Filter（中值濾波）
            # median = cv2.medianBlur(img, 5)

            # # 方法三：Bilateral Filter（雙邊濾波）
            # bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

            # 方法四：Non-Local Means Denoising（非局部均值降噪）
            nlmeans = cv2.fastNlMeansDenoisingColored(img, None, h=15, hColor=15, templateWindowSize=7, searchWindowSize=21)

            # TODO 輸出模組化
            output_image_name = "preprocessed_image.jpg"
            output_image_path = save_preprocessed_image(nlmeans, output_image_name)

            print("降噪處理完成，圖片已儲存至：", f'<img {output_image_path}>')
            return "降噪處理完成，圖片已儲存至：" + f'<img {output_image_path}>'

        
        def image_correct(image_name: str) -> str:
            """Correct the tilt (skew) of the input image using edge and Hough line detection."""

            image_path = os.path.join("image", image_name)
            img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)

            # 邊緣偵測（Canny）
            edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)

            # 霍夫直線轉換（Hough Line Detection）
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
            if lines is None:
                print("無法偵測到直線，圖片未執行校正。")
                return "無法偵測到直線，圖片未執行校正。"

            # 計算平均傾斜角度
            angles = []
            for rho, theta in lines[:, 0]:
                angle = (theta * 180 / np.pi) - 90
                angles.append(angle)
            avg_angle = np.mean(angles)
            print(f"偵測到傾斜角度：約 {avg_angle:.2f} 度")

            # 執行旋轉校正
            (h, w) = img_color.shape[:2]
            center = (w // 2, h // 2)
            rot_mat = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
            corrected = cv2.warpAffine(img_color, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

            # 儲存校正後圖片
            output_image_name = "corrected_" + image_name
            output_image_path = os.path.join("preprocessed_images", output_image_name)
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            cv2.imwrite(output_image_path, corrected)

            print("圖片已校正並儲存至：", f'<img {output_image_path}>')
            return "圖片已校正並儲存至：" + f'<img {output_image_path}>'

        def image_roi(rel_bbox: str) -> str:
            """
            根據相對座標從圖片中擷取ROI，並儲存為新圖片。
            :param rel_bbox: 相對座標 [x1, y1, x2, y2]，皆為 0~1 範圍
            :return: 儲存的圖片路徑
            """

            folder_path, input_image_name = get_input_image()
            input_image_path = os.path.join(folder_path, input_image_name)
            img = cv2.imread(input_image_path)

            rel_bbox = ast.literal_eval(rel_bbox)  # 將字串轉為列表
            h, w, _ = img.shape  # 獲取圖片的寬和高
            x1, y1, x2, y2 = [int(rel_bbox[i] * w) if i % 2 == 0 else int(rel_bbox[i] * h) for i in range(4)]
            roi = img[y1:y2, x1:x2]  # 使用 NumPy 切片擷取 ROI

            output_image_name = "preprocessed_image.jpg"
            output_image_path = save_preprocessed_image(roi, output_image_name)

            print("ROI擷取完成，圖片已儲存至：", f'<img {output_image_path}>')
            return "ROI擷取完成，圖片已儲存至：" + f'<img {output_image_path}>'

        self.tool_list = {
            "image_denoise": image_denoise,
            "image_correct": image_correct,
            "image_roi": image_roi,
            "get_input_image": get_input_image,
            "save_preprocessed_image": save_preprocessed_image,
        }

    def register_tool(self, agents, tool_names, caller_name, executor_name):
        for tool_name in tool_names:
            tool = self.tool_list[tool_name]
            func_name = tool.__name__
            func_description = inspect.getdoc(tool)
            register_function(
                tool,
                caller=agents[caller_name],  # The assistant agent can suggest calls to the calculator.
                executor=agents[executor_name],  # The user proxy agent can execute the calculator calls.
                name=func_name,  # By default, the function name is used as the tool name.
                description=func_description,  # A description of the tool.
            )