import tkinter as tk
from tkinter import ttk
import images_function_tools as ift

class StackFrame:
    """
    This class will be used to accumulate all the operations on an image
    each stack frae represent an operation
    """
    def __init__(self,img:ift.MyImage,f:callable,args:tuple) -> None:
        self.function = f
        self.img_in = img
        self.args = args
        r = self.function(self.img_in,*self.args)
        if isinstance(r,list):
            self.imgs_out:list[ift.MyImage] = r
        elif isinstance(r,ift.MyImage):
            self.imgs_out:list[ift.MyImage] = [r]

class Application(tk.Tk):
    """
    The application will be initialized by the init , inside the innit several calls for deferent init function will occure , to create the components the layout and other stuff
    At the end a binding function will be called to define behaviours
    """
    def __init__(self):
        super().__init__()
        self.title("Image processing application")
        self.geometry("400x500")
        self.__create_components__()

    def __create_components__(self):
        self.__create__menu_bare__()


    def __create_layout__(self):
        ...

    def __create__menu_bare__(self):
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
    ...