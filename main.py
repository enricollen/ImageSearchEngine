import tkinter as tk
from user_interface import UserInterface
from image_searcher import ImageSearcher

if __name__ == "__main__":
    
    root = tk.Tk()
    searcher = ImageSearcher()
    gui = UserInterface(root, searcher)
    root.mainloop()