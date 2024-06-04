import torch
import clip
import pickle

class ImageSearcher:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.embeddings, self.paths = self.load_embeddings()

    def load_embeddings(self):
        with open('image_embeddings.pkl', 'rb') as f:
            data = pickle.load(f)
        return data['embeddings'], data['paths']

    def find_similar_images(self, text_description, number_of_images):
        text = clip.tokenize([text_description]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
            embeddings_norm = self.embeddings / self.embeddings.norm(dim=1, keepdim=True)
            text_features_norm = text_features / text_features.norm(dim=1, keepdim=True)
            similarities = (text_features_norm @ embeddings_norm.T).squeeze(0)
            best_indices = similarities.argsort(descending=True)[:number_of_images]
        adjusted_similarities = (similarities + 1) / 2
        return [(self.paths[i], adjusted_similarities[i].item(), self.embeddings[i]) for i in best_indices], text_features