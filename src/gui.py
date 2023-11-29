import tkinter as tk
from tkinter import ttk
from ttkbootstrap.window import Window
import images_function_tools as ift

class Application(Window):
    """
    The application will be initialized by the init , inside the innit several calls for deferent init function will occure , to create the components the layout and other stuff
    At the end a binding function will be called to define behaviours
    """

    def __init__(self):
        super().__init__()
        self.theme = "superhero"
        self.title("Image processing software")
        self.geometry("1100x700")
        self.__create_components__()

    def __create_components__(self):
        self.__create__menu_bare__()
        self.__create_operations_stack__()
    def __create_layout__(self):
        ...

    def __create__menu_bare__(self):
        menu = tk.Menu(self)
        menues: dict[str:tk.Menu] = {}
        self.configure(menu=menu)
        categories = ["Files", "Geometric operations", "Photometric operation", "Filters", "Histogram based operations",
                      "Segmentation", "Statistical constants", "Visualization"]
        
        for m in categories:
            sub_menu = tk.Menu(menu, tearoff=False)
            menu.add_cascade(label=m, menu=sub_menu)
            menues[m] = sub_menu

        # the Files sub menu
        menues["Files"].add_command(label='Open Image', command=None)
        menues["Files"].add_command(label='Save Image', command=None)
        menues["Files"].add_separator()
        menues["Files"].add_command(label='Exit', command=None)

        menues["Geometric operations"].add_command(label='Translation', command=None)
        menues["Geometric operations"].add_command(label='Rotation', command=None)
        menues["Geometric operations"].add_command(label='Reflection', command=None)
        menues["Geometric operations"].add_command(label='Re-scale', command=None)
        menues["Geometric operations"].add_command(label='Cut', command=None)
        menues["Geometric operations"].add_command(label='Past on canvas', command=None)
        menues["Geometric operations"].add_command(label='Overlay', command=None)

        menues["Photometric operation"].add_command(label='Gray scale', command=None)
        menues["Photometric operation"].add_command(label='Resolution under-scaling', command=None)

        menues["Filters"].add_command(label='Mean', command=None)
        menues["Filters"].add_command(label='Gaussian', command=None)
        menues["Filters"].add_command(label='Bilateral', command=None)
        menues["Filters"].add_command(label='Median', command=None)
        menues["Filters"].add_command(label='Lablacian', command=None)

        menues["Histogram based operations"].add_command(label='Translation', command=None)
        menues["Histogram based operations"].add_command(label='Inverse', command=None)
        menues["Histogram based operations"].add_command(label='Dynamic expansion', command=None)
        menues["Histogram based operations"].add_command(label='Equalization', command=None)
        menues["Histogram based operations"].add_command(label='Histogram matching', command=None)

        menues["Segmentation"].add_command(label='Clustering', command=None)
        menues["Segmentation"].add_command(label='Tagging', command=None)
        menues["Segmentation"].add_command(label='Edge detection', command=None)

        menues["Statistical constants"].add_command(label='Mean', command=None)
        menues["Statistical constants"].add_command(label='Std', command=None)
        menues["Statistical constants"].add_command(label='Variance', command=None)
        menues["Statistical constants"].add_command(label='Median', command=None)
        menues["Statistical constants"].add_command(label='Outliers', command=None)

        menues["Visualization"].add_command(label='Show Image', command=None)
        menues["Visualization"].add_command(label='Show Histogram', command=None)
        menues["Visualization"].add_command(label='Show Normalized Histogram', command=None)
        menues["Visualization"].add_command(label='Show Cumulative Histogram', command=None)
        menues["Visualization"].add_command(label='Show Cumulative Normalized Histogram', command=None)

    def __create_operations_stack__(self):
        self.operation_stack_tree_view = ttk.Treeview(master=self, columns=('N°', 'Operation', 'Args'), show='headings')

        self.operation_stack_tree_view.heading('N°', text='N°')
        self.operation_stack_tree_view.heading('Operation', text='Operation')
        self.operation_stack_tree_view.heading('Args', text='Args')

        self.operation_stack_tree_view.pack(expand=True, fill=tk.BOTH)
    def run(self):
        self.mainloop()


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


if __name__ == "__main__":
    Application().run()