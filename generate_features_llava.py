import os
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from sentence_transformers import SentenceTransformer
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"
meme_folder = r"C:\Users\20578\Desktop\pose_memes_match\static\memes"
output_dir = "llava_features"
os.makedirs(output_dir, exist_ok=True)

print("Loading LLaVA model...")
model_id = "llava-hf/llava-1.5-7b-hf"
processor = LlavaProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)

print("Loading SentenceTransformer...")
text_encoder = SentenceTransformer("all-MiniLM-L6-v2")

meme_features = []
meme_captions = []
meme_names = []

for filename in tqdm(os.listdir(meme_folder)):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    image_path = os.path.join(meme_folder, filename)
    image = Image.open(image_path).convert("RGB")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this meme in one sentence, including its emotion and situation:"},
                {"type": "image"},
            ],
        }
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=80)
        caption = processor.decode(output[0], skip_special_tokens=True)

    meme_captions.append(caption)
    meme_names.append(filename)
    embedding = text_encoder.encode(caption, normalize_embeddings=True)
    meme_features.append(embedding)

np.save(os.path.join(output_dir, "meme_features.npy"), np.array(meme_features))
with open(os.path.join(output_dir, "meme_captions.pkl"), "wb") as f:
    pickle.dump({"names": meme_names, "captions": meme_captions}, f)

print("✅ Features generated and saved in", output_dir)

