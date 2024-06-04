import torch
import clip
from PIL import Image
import os
import pickle


class Indexer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def load_and_preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return self.preprocess(image).unsqueeze(0).to(self.device)

    def create_image_embeddings(self, directory):
        image_paths = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith(('.png', '.jpg', '.jpeg'))]
        image_embeddings = []
        names = []

        for path in image_paths:
            image = self.load_and_preprocess_image(path)
            with torch.no_grad():
                embedding = self.model.encode_image(image)
            image_embeddings.append(embedding)
            names.append(path)

        # save both the embeddings and paths
        with open('image_embeddings.pkl', 'wb') as f:
            pickle.dump({'embeddings': torch.cat(image_embeddings), 'paths': names}, f)


#index = Indexer()
#index.create_image_embeddings("img/")