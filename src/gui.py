import tkinter as tk
from tkinter import ttk
from ttkbootstrap.window import Window
import images_function_tools as ift
from tkinter import filedialog,messagebox,simpledialog
import os
import sys

class StackFrame:
    """
    This class will be used to accumulate all the operations on an image
    each stack frae represent an operation
    """
    def __init__(self,img:ift.MyImage,f:callable,args:tuple,_type:str) -> None:
        """
        _type : must be s or o (s for static and o for objetc) 
        """
        _type = _type.lower()
        if _type not in ('s','o'):raise ValueError(f'the selected type {_type} is not correct')

        self.function = f
        self.img_in = img
        self.args = args
        self._type = _type

        if self._type == 'o':
            r = self.function(self.img_in,*self.args)
        
        elif self._type == 's':
            r = self.function(*self.args)

        else:raise Exception(f"type {_type} is not supported")

        if not isinstance(r,list) and not isinstance(r,ift.MyImage):
            raise Exception("The only types accapted as output for the stack frame are list of MyImage or MyImage")
        
        if isinstance(r,ift.MyImage):r = [r]
        self.imgs_out:list[ift.MyImage] = r 


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

        self.operation_stack:list[StackFrame] = []

    def __create_components__(self):
        self.__create__menu_bare__()
        self.__create_operations_stack__()
        self.__create_buttons__()
    

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
        menues["Files"].add_command(label='Open Image', command=self.open_image_menu_bare_command)
        menues["Files"].add_command(label='Save Image', command=self.save_image_menu_bare_command)
        menues["Files"].add_separator()
        menues["Files"].add_command(label='Exit', command=self.exit_menu_bare_command)

        menues["Geometric operations"].add_command(label='Translation', command=self.geometric_operations_translation_menu_bare_command)
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

        menues["Visualization"].add_command(label='Show Image', command=self.visualization_show_menu_bare_command)
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
    
    def __create_buttons__(self):
        self.undo_btn = ttk.Button(self,text="Undo",command=None)

        self.undo_btn.pack(fill=tk.BOTH)

    def run(self):
        self.mainloop()

    # menu bare function
    def open_image_menu_bare_command(self):
        path = filedialog.askopenfilename()
        if path is None or  len(path) == 0:
            messagebox.showerror("ERROR","PROVIDE A CORRECT PATH TO THE IMAGE")
            return
        
        img = None
        try:
            img = ift.MyImage.open_image_as_rgb_matrices(path)
        except Exception as e:
            print(e)
            messagebox.showerror("ERROR","UNCORRECT PATH")
            return
        
        self.operation_stack.append(StackFrame(None,ift.MyImage.open_image_as_rgb_matrices,(path,),'s'))

        self.redraw_operation_stack_tree_view()

    def save_image_menu_bare_command(self):
        """
        - test if the stack frame is not empty
        - ask for the path and the name of the image
        - remove any extension from the name of the image
        - save the image
        - if any exception happen show an error message to the user
        """
        if len(self.operation_stack) == 0:
            messagebox.showerror("ERROR","NO IMAGE TO SAVE")
            return
        
        # testing the path
        path = filedialog.askdirectory()
        if path is None or not path:
            messagebox.showerror("EROOR","PATH IS NOT CORRECT")
            return
        path = path.strip()
        
        # image name
        image_name = simpledialog.askstring("INPUT","ENTER THE IMAGE NAME")
        if image_name is not None :
            image_name = image_name.strip()
        
        if image_name is None or image_name == "":
            messagebox.showerror('ERROR',"PROVIDE A CORRECT IMAGE NAME")
            return
        
        extension = ""
        for e in ".png",".jpeg",".jpg":
            if image_name.lower().endswith(e):
                break
        if extension == "": extension = ".png"
        
        imgs:list[ift.MyImage] = self.get_selected_images()
        if len(imgs) == 0: messagebox.showerror("ERROR","NO IMAGE WASD PROVIDED")
        try:
            for i,img in enumerate(imgs):
                # if not extension was provided add png by default
                imgi_name = os.path.join(path,image_name+str(i)+extension)
                img.save_image(imgi_name)
        
        except Exception as e:
            messagebox.showerror("ERROR",f"CAN'T SAVE IMAGE")
            print(type(e))
            return
        x = "s" if len(imgs) > 1 else ""
        messagebox.showinfo("INFO",f"IMAGE{x} WERE SAVED")
    
    def exit_menu_bare_command(self):
        if messagebox.askyesno("CONFIRMATION","WOULD YOU LIKE TO EXIT THE APPLICATION"):
            sys.exit(0)
    
    # GEOMETRIC OPERATION
    def geometric_operations_translation_menu_bare_command(self):
        """
        - test if the stack frame is not empty
        - ask for how many pixels to translate for the x axes and the y axes
        - perfomre the operation
        - push into the stack the result as a StackFrame
        """
        if len(self.operation_stack) == 0:
            messagebox.showerror("ERROR","NO IMAGE WAS PROVIDED")
            return
        
        input_imgs =  self.operation_stack[-1].imgs_out
        if len(input_imgs) <= 0:
            messagebox.showerror("ERROR","ERROR NO IMAGE WAS FOUND THIS ERROR SHOUD NEVER HAPPEN CONTACT DEVS")
            return
        
        if len(input_imgs) > 1:
            messagebox.showerror("ERROR","THE LAST OPERATION GENERATED MORE THAN ONE IMAGES PLEASE COMBINE THEM USING OVERLAYING OR DISCARD THEM")
            return
        input_imgs = input_imgs[0]

        x = simpledialog.askinteger("TRANSLATION","ENTER NUMBER OF PXLS FOR TRANSLATION OF THE X AXES")
        if x == None:
            messagebox.showerror("ERROR","OPERATION WAS CANCELED")
            return
        
        y = simpledialog.askinteger("TRANSLATION","ENTER NUMBER OF PXLS FOR TRANSLATION OF THE Y AXES")
        if y == None:
            messagebox.showerror("ERROR","OPERATION WAS CANCELED")
            return
        
        self.operation_stack.append(StackFrame(input_imgs,ift.MyImage.translation,((x,y),),'o'))
        self.redraw_operation_stack_tree_view()
    
    def geometric_operations_rotation_menu_bare_command(self):
        """
        - test if the stack frame is not empty
        - ask how many degrees to rotate
        - performe operation
        - push into the stack the result as a StackFrame
        """
        if len(self.operation_stack) == 0:
            messagebox.showerror("ERROR","NO IMAGE WAS PROVIDED")
            return
        
        input_imgs =  self.operation_stack[-1].imgs_out
        if len(input_imgs) <= 0:
            messagebox.showerror("ERROR","ERROR NO IMAGE WAS FOUND THIS ERROR SHOUD NEVER HAPPEN CONTACT DEVS")
            return
        
        if len(input_imgs) > 1:
            messagebox.showerror("ERROR","THE LAST OPERATION GENERATED MORE THAN ONE IMAGES PLEASE COMBINE THEM USING OVERLAYING OR DISCARD THEM")
            return
        input_imgs = input_imgs[0]
        
        theta = simpledialog.askinteger("TRANSLATION","ENTER ANGLE IN DEGREES")
        if theta is None:
            messagebox.showerror("ERROR","OPERATION WAS CANCELED")
            return
        
        self.operation_stack.append(StackFrame(input_imgs,ift.MyImage.rotate,(theta,),'o'))
        self.redraw_operation_stack_tree_view()


    # VISUALISATION
    def visualization_show_menu_bare_command(self):
        """
        - test if the stack is empty
        """
        if len(self.operation_stack) == 0:
            messagebox.showerror('ERROR',"NO IMAGE TO SHOW")
            return
        imges = self.get_selected_images()
        if len(imges) == 0:
            messagebox.showerror("ERROR","NO IMAGE WAS PROVIDED")
            return
        ift.plt.clf()
        ift.MyImage.show_images(imges)
    

    # useful methods
    def redraw_operation_stack_tree_view(self):
        self.operation_stack_tree_view.delete(*self.operation_stack_tree_view.get_children())
        for i,stck in enumerate(self.operation_stack):
            v = (str(i),stck.function.__name__,str(stck.args))
            self.operation_stack_tree_view.insert("","end",values=v)
    
    def get_selected_images(self) -> list[ift.MyImage]:
        imges = []
        selected_stack_frames = self.operation_stack_tree_view.selection()
        for i in selected_stack_frames:
            j = int(self.operation_stack_tree_view.item(i)["values"][0])
            imges.extend(self.operation_stack[j].imgs_out)
        return imges

if __name__ == "__main__":
    Application().run()