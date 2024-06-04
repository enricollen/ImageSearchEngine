# Simple Image Search Engine Using CLIP 🖼️

This project demonstrates a practical application of the ([CLIP](https://arxiv.org/abs/2103.00020)) (Contrastive Language–Image Pre-training) model by OpenAI, used to create embeddings for both images and textual descriptions.
This model, in fact, allows to project images and text into the same latent space, enabling the creation of a simple image search engine starting from a textual query. 

## Project Overview 🔍
As first step is required to perform the indexing of a collection of images inside a folder; this process will produce an embedding for each image in the folder. 
These embeddings are then compared to the embedding of a textual description provided by the user. 
By computing the cosine similarity between the image embeddings and the text embedding, the system can identify and return the images most relevant to the textual description.

The top-k images are then showed in descending order of similarity w.r.t. textual query.

### Key Features:
- **Image Embedding Generation**: Utilize the CLIP model to generate embeddings for each image in a collection.
- **Data Storage**: For simplicity, embeddings are stored in a `.pkl` file using `pickle`. This method is for sake of simplicity but easy adaptable to vector databases for handling larger image collections.
- **Text-to-Image Search**: Whenever the user enters a text query, the system finds the most similar images based on the cosine similarity between CLIP-generated image embeddings and text embedding.
- **Interactive GUI**: A simple Tkinter-based GUI that allows users to input their queries and view the results.
- **Embedding Visualization**: Functionality to visualize the embeddings in 3D (using PCA), showing how images and text relate in the embedded space.

## 🛠️ Setup and Local Deployment
- Clone the repository and navigate to the project directory.
- Install dependencies: `pip install -r requirements.txt`
- Run the application: `python main.py`

## Screenshots 📸
Here is a screenshot illustrating the simple search engine interface:

<img src="https://github.com/enricollen/ImageSearchEngine/blob/main/example/example_1.jpg" width="400" height="400">
<img src="https://github.com/enricollen/ImageSearchEngine/blob/main/example/example_3.jpg"  width="800" height="400">
<img src="https://github.com/enricollen/ImageSearchEngine/blob/main/example/example_2.jpeg"  width="400" height="400">
