import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class UserInterface:

    def __init__(self, master, searcher):
        self.master = master
        self.searcher = searcher
        self.master.title("Image Search")
        self.create_widgets()

    def create_widgets(self):
        entry_frame = tk.Frame(self.master)
        entry_frame.pack(pady=20)

        self.entry = tk.Entry(entry_frame, width=40)
        self.entry.pack(side=tk.LEFT, padx=(0, 10))

        self.top_images_var = tk.StringVar(value='3')
        combobox = ttk.Combobox(entry_frame, textvariable=self.top_images_var, width=3)
        combobox['values'] = [3, 5, 10, 15, 20]
        combobox.current(0)
        combobox.pack(side=tk.LEFT)

        search_button = tk.Button(self.master, text="Search", command=self.search_images)
        search_button.pack(pady=10)

        plot_button = tk.Button(self.master, text="Plot Embeddings", command=self.plot_embeddings)
        plot_button.pack(pady=10)

        self.frame = tk.Frame(self.master)
        self.frame.pack(padx=10, pady=10, expand=True)

    def search_images(self):
        description = self.entry.get()
        number_of_images = int(self.top_images_var.get())

        if not description:
            messagebox.showinfo("Input needed", "Please enter a description.")
            return None, None 

        result_paths, text_features = self.searcher.find_similar_images(description, number_of_images)
        if not result_paths:  # no results are found
            messagebox.showinfo("Result", "No similar images found.")
            return None, None

        self.update_image_grid(result_paths)
        return result_paths, text_features


    def update_image_grid(self, image_results):
        for widget in self.frame.winfo_children():
            widget.destroy()

        for i, (path, score, _) in enumerate(image_results):
            img = Image.open(path).resize((150, 150), Image.Resampling.LANCZOS)
            img = ImageTk.PhotoImage(img)
            panel = tk.Label(self.frame, image=img)
            panel.image = img 
            panel.grid(row=0, column=i, padx=5, pady=5)
            label = tk.Label(self.frame, text=f"Similarity: {score:.2f}")
            label.grid(row=1, column=i, padx=5, pady=5)

    def plot_embeddings(self):
        description = self.entry.get()
        if not description:
            messagebox.showinfo("Input needed", "Please enter a description.")
            return

        image_results, text_features = self.search_images()
        if image_results is None or text_features is None: 
            return  # exit the function if there's nothing to plot

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
            ax.text(x + 0.02, y + 0.02, z + 0.02, label, fontsize=9)

        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_zlabel('PCA Component 3')
        ax.set_title('3D Plot of Embeddings')
        plt.show()