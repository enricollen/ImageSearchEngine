import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import torch
import clip
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def load_embeddings():
    with open('image_embeddings.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['embeddings'], data['paths']

embeddings, paths = load_embeddings()

def find_similar_images(text_description, embeddings, paths):
    number_of_images = int(top_images_var.get())
    text = clip.tokenize([text_description]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)
        embeddings_norm = embeddings / embeddings.norm(dim=1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=1, keepdim=True)
        similarities = (text_features_norm @ embeddings_norm.T).squeeze(0)
        best_indices = similarities.argsort(descending=True)[:number_of_images]

    # remapping similarities from [-1, 1] to [0, 1]
    adjusted_similarities = (similarities + 1) / 2

    return [(paths[i], adjusted_similarities[i].item(), embeddings[i]) for i in best_indices], text_features

def search_images():
    description = entry.get()
    if not description:
        messagebox.showinfo("Input needed", "Please enter a description.")
        return
    
    result_paths, text_features = find_similar_images(description, embeddings, paths)
    update_image_grid(result_paths)
    return result_paths, text_features

def update_image_grid(image_results):
    for widget in frame.winfo_children():
        widget.destroy()

    for i, (path, score, _) in enumerate(image_results):
        img = Image.open(path)
        img = img.resize((150, 150), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(img)
        
        # image in the grid
        panel = tk.Label(frame, image=img)
        panel.image = img  # Keep a reference to avoid garbage collection
        panel.grid(row=0, column=i, padx=5, pady=5)  

        # similarity score below each image
        label = tk.Label(frame, text=f"Similarity: {score:.2f}")
        label.grid(row=1, column=i, padx=5, pady=5) 

def plot_embeddings():
    description = entry.get()
    if not description:
        messagebox.showinfo("Input needed", "Please enter a description.")
        return

    image_results, text_features = search_images()

    embeddings_list = [embedding.cpu().numpy() for _, _, embedding in image_results]
    embeddings_list.append(text_features.squeeze(0).cpu().numpy())
    embeddings_array = np.array(embeddings_list)
    labels = [f"Image {i+1}" for i in range(len(image_results))] + ["Text"]

    perplexity = min(30, len(embeddings_list) - 1)
    tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity)
    embeddings_3d = tsne.fit_transform(embeddings_array)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    for i, label in enumerate(labels):
        x, y, z = embeddings_3d[i]
        ax.scatter(x, y, z)
        ax.text(x+0.02, y+0.02, z+0.02, label, fontsize=9)
    
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    ax.set_title('3D Plot of Embeddings')
    plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Image Search")

    entry_frame = tk.Frame(root)
    entry_frame.pack(pady=20)

    entry = tk.Entry(entry_frame, width=40)
    entry.pack(side=tk.LEFT, padx=(0, 10))

    top_images_var = tk.StringVar(value='3')  # Default number of images to show
    combobox = ttk.Combobox(entry_frame, textvariable=top_images_var, width=3)
    combobox['values'] = [3, 5, 10, 15, 20]  
    combobox.current(0)
    combobox.pack(side=tk.LEFT)

    search_button = tk.Button(root, text="Search", command=search_images)
    search_button.pack(pady=10)

    plot_button = tk.Button(root, text="Plot Embeddings", command=plot_embeddings)
    plot_button.pack(pady=10)

    frame = tk.Frame(root)
    frame.pack(padx=10, pady=10, expand=True)

    root.mainloop()