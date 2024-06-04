import torch
import clip
from PIL import Image
import os
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0).to(device)

def create_image_embeddings(directory):
    image_paths = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith(('.png', '.jpg', '.jpeg'))]
    image_embeddings = []
    names = []

    for path in image_paths:
        image = load_and_preprocess_image(path)
        with torch.no_grad():
            embedding = model.encode_image(image)
        image_embeddings.append(embedding)
        names.append(path)

    # save both the embeddings and paths
    with open('image_embeddings.pkl', 'wb') as f:
        pickle.dump({'embeddings': torch.cat(image_embeddings), 'paths': names}, f)

if __name__ == "__main__":
    create_image_embeddings('./img')