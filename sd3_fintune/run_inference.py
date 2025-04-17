import torch
from diffusers import DiffusionPipeline
import torch
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
import os
import time  # 


# 設定環境變數
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.cuda.empty_cache()
seed_value=42
generator=torch.Generator().manual_seed(seed_value)

pipe_id = "stabilityai/stable-diffusion-3-medium-diffusers"

# 記錄 LoRA 加載開始時間
start_time = time.time()

pipe = DiffusionPipeline.from_pretrained(
    pipe_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True
).to("cuda")

pipe.enable_model_cpu_offload()



# pipe = DiffusionPipeline.from_pretrained(pipe_id, torch_dtype=torch.float16).to("cuda")
pipe.load_lora_weights("YOUR_OUTPUT_PATH", weight_name="MODEL_WEIGHT_NAME",adapter_name="trained")
pipe.fuse_lora(adapter_names=["trained"], lora_scale=0.8)

# 記錄 LoRA 加載時間
lora_load_time = time.time() - start_time
print(f"LoRA 加載時間: {lora_load_time:.2f} 秒")

prompt = "The square grey metallic lid of the crystal oscillator appears worn, with fine surface scratches and a reflective silver border."

# 指定儲存路徑
save_dir = "SAVE_DIR"  
os.makedirs(save_dir, exist_ok=True)  # 如果資料夾不存在則建立

batch_size = 10  # 每次生成 10 張
total_images = 100  # 總共要生成 100 張

# 記錄圖片生成開始時間
generation_start_time = time.time()

for batch in range(total_images // batch_size):  # 執行 100/10 = 10 次
    generator = torch.Generator().manual_seed(seed_value + batch)  # 使用不同 seed

    batch_start_time = time.time()  # 記錄 batch 開始時間

    images = pipe(
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=28,
        height=512,
        width=512,
        guidance_scale=7.0,
        generator=generator,
        num_images_per_prompt=batch_size  # 每次生成 10 張
    ).images

    # 存檔
    for i, img in enumerate(images):
        img_index = batch * batch_size + i  # 計算總索引
        img.save(os.path.join(save_dir, f"0209_v100_{img_index}.png"))

    batch_time = time.time() - batch_start_time  # 計算 batch 執行時間
    print(f"Batch {batch + 1} saved ({(batch + 1) * batch_size}/{total_images} images) - 花費 {batch_time:.2f} 秒")

    # print(f"Batch {batch + 1} saved ({(batch + 1) * batch_size}/{total_images} images)")

print(f"All {total_images} images saved to {save_dir}")

# 記錄總圖片生成時間
total_generation_time = time.time() - generation_start_time
print(f"所有 {total_images} 張圖片生成完畢，總共花費 {total_generation_time:.2f} 秒")


