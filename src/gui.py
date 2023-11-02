import tkinter as tk
from tkinter import ttk


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image processing software")
        self.geometry("400x500")
        self.__add_components__()

    def __add_components__(self):
        self.__add__menu_bare__()

    def __add__menu_bare__(self):
        menu = tk.Menu(self)
        menues:dict[str:tk.Menu] = {}
        self.configure(menu=menu)

        for m in ["Files","Image editing","Segmantation","Image analysing"]:
            sub_menu = tk.Menu(menu,tearoff=False)
            menu.add_cascade(label=m,menu=sub_menu)
            menues[m] = sub_menu
        
        # the Files sub menu
        menues["Files"].add_command(label='Open Image',command=None)
        menues["Files"].add_command(label='Save Image',command=None)
        menues["Files"].add_separator()
        menues["Files"].add_command(label='Exit',command=None)


    def run(self):
        self.mainloop()


if __name__ == "__main__":
    Application().run()
    # test