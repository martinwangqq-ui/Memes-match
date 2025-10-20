import torch
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, jsonify
from transformers import AutoProcessor, AutoModelForVision2Seq
from sentence_transformers import SentenceTransformer
import pickle, os

app = Flask(__name__)

# ----------------------------
# 1. 路径设置
# ----------------------------
# uploads 放在 static/uploads 目录
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ----------------------------
# 2. 加载模型与特征
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "llava-hf/llava-1.5-7b-hf"

print(">>> 正在加载模型，这可能需要几分钟...")
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)

text_encoder = SentenceTransformer("all-MiniLM-L6-v2")

# 载入预提取的 meme 特征
meme_features = np.load("llava_features/meme_features.npy")
with open("llava_features/meme_captions.pkl", "rb") as f:
    meme_info = pickle.load(f)

print(">>> 模型加载完成 ✅")

# ----------------------------
# 3. Meme 匹配逻辑
# ----------------------------
def match_meme(image_path):
    image = Image.open(image_path).convert("RGB")

    # 必须带 <image> token
    prompt = "<image>\nDescribe this meme-like image in one sentence, focusing on its emotion and situation:"

    # 注意参数顺序：text=prompt, images=image
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=80)

    caption = processor.decode(output[0], skip_special_tokens=True)

    # 用 sentence-transformer 做 embedding 匹配
    query_emb = text_encoder.encode(caption, normalize_embeddings=True)
    similarities = np.dot(meme_features, query_emb)
    idx = np.argmax(similarities)

    return {
        "caption": caption,
        "match_name": meme_info["names"][idx],
        "match_caption": meme_info["captions"][idx],
        "score": float(similarities[idx])
    }


# ----------------------------
# 4. Flask 路由
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]

    # ✅ 保存到 static/uploads 下
    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    # ✅ 调用模型推理
    result = match_meme(save_path)

    # ✅ 返回 JSON + 图片路径（网页端可显示）
    result["user_image"] = f"/static/uploads/{file.filename}"

    return render_template("result.html", result=result)

# ----------------------------
# 5. 启动 Flask
# ----------------------------
if __name__ == "__main__":
    print(">>> Flask 启动中，请访问 http://127.0.0.1:5000/")
    app.run(host="0.0.0.0", port=5000, debug=True)
