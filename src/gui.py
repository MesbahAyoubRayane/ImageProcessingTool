import tkinter as tk
from tkinter import ttk


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image processing software")
        self.geometry("400x500")

        self.__add_components__()

    def __add_components__(self):
         ...

    def __add__menu_bare__(self):
        menu = tk.Menu(self)
        menues:list[tk.Menu] = ["Files","Image editing","Segmantation","Image analysing"]


        self.configure(menu=menu)

    def run(self):
        self.mainloop()


if __name__ == "__main__":
    Application().run()