import tkinter as tk
from tkinter import ttk
#from ttkbootstrap.window import Window
from ttkthemes import ThemedTk
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

class MetaDataFrame(tk.Toplevel):
    def __init__(self,master:tk.Frame|tk.Tk,imgs:list[ift.MyImage]):
        super().__init__(master=master)
        for i,img in enumerate(imgs):
            frm = ttk.Frame(self)
            
            ttk.Label(frm,text="IMAGE N°: "      + str(i)).pack(side=tk.TOP,fill=tk.BOTH,expand=True)
            ttk.Label(frm,text="DIMENSIONS : "   + str(img.dimensions)).pack(side=tk.TOP,fill=tk.BOTH,expand=True)
            ttk.Label(frm,text="IMAGE MODE : "   + str(img.mode)).pack(side=tk.TOP,fill=tk.BOTH,expand=True)
            mean = img.mean()
            if isinstance(mean,tuple):
                mean = round(mean[0],2),round(mean[1],2),round(mean[2],2)
            else:
                mean = round(mean,2)
            ttk.Label(frm,text="IMAGE MEAN : "   + str(mean)).pack(side=tk.TOP,fill=tk.BOTH,expand=True)
            median = img.median()
            if isinstance(median,tuple):
                median = round(median[0],2),round(median[1],2),round(median[2],2)
            else:
                median = round(median,2)
            ttk.Label(frm,text="IMAGE MEDIAN : " + str(median)).pack(side=tk.TOP,fill=tk.BOTH,expand=True)

            std = img.std()
            if isinstance(std,tuple):
                std = round(std[0],2),round(std[1],2),round(std[2],2)
            else:
                std = round(std,2)
            ttk.Label(frm,text="IMAGE STANDARE DEVIATION : " + str(std)).pack(side=tk.TOP,fill=tk.BOTH,expand=True)
            
            out = img.outliers()
            if isinstance(out,tuple):
                out = sum(out) / (img.width * img.height * 3)
            elif isinstance(out,int):
                out = out / (img.width * img.height)
            out = round(out * 100,2)
            ttk.Label(frm,text="IMAGE OUTLIERS : "+ str(out)+"%").pack(side=tk.TOP,fill=tk.BOTH,expand=True)

            frm.pack(side=tk.LEFT,fill=tk.BOTH,expand=True,padx=10,pady=10)
        
        self.resizable(False,False)
    
    def run(self):
        self.mainloop()

class Application(ThemedTk):

    """
    The application will be initialized by the init , inside the innit several calls for deferent init function will occure , to create the components the layout and other stuff
    At the end a binding function will be called to define behaviours
    """

    def __init__(self):
        super().__init__(theme='breeze') 
        self.title("Image processing software")
        self.geometry("1100x700")
        self.__create_components__()

        # gloabl variables
        self.meta_data_frame:MetaDataFrame = None

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
                      "Segmentation", "Visualization"]
        
        for m in categories:
            sub_menu = tk.Menu(menu, tearoff=False)
            menu.add_cascade(label=m, menu=sub_menu)
            menues[m] = sub_menu

        # the Files sub menu
        menues["Files"].add_command(label='Open Image', command=self.open_image_menu_bare_command) # 0 -> 1
        menues["Files"].add_command(label='Save Image', command=self.save_image_menu_bare_command) # 1 -> 0
        menues["Files"].add_separator()
        menues["Files"].add_command(label='Exit', command=self.exit_menu_bare_command)

        menues["Geometric operations"].add_command(label='Translation', command=self.geometric_operations_translation_menu_bare_command) # 1 -> 1
        menues["Geometric operations"].add_command(label='Rotation', command=self.geometric_operations_rotation_menu_bare_command) # 1 -> 1
        menues["Geometric operations"].add_command(label='Reflection', command=self.geometric_operations_reflection_menu_bare_command) # 1 -> 1
        menues["Geometric operations"].add_command(label='Re-scale', command=self.geometric_operations_rescale_menu_bare_command) # 1 -> 1
        menues["Geometric operations"].add_command(label='Cut', command=self.geometric_operatioins_cut_menu_bare_command) # 1 -> 1
        menues["Geometric operations"].add_command(label='Past on canvas', command=self.geometric_operatioins_past_on_canvas_menu_bare_command) # 1 -> 1
        menues["Geometric operations"].add_command(label='Overlay', command=self.geometric_operatioins_overlay_menu_bare_command) # n -> 1

        menues["Photometric operation"].add_command(label='Gray scale', command=self.photometric_operation_gray_scale_menu_bare_command) # 1 -> 1
        menues["Photometric operation"].add_command(label='Resolution under-scaling', command=self.photometric_operation_resolution_under_scaling) # 1 -> 1

        menues["Filters"].add_command(label='Mean', command=self.filters_mean_menu_bare_command) # 1 -> 1
        menues["Filters"].add_command(label='Gaussian', command=self.filters_gaussian_menu_bare_command) # 1 -> 1
        menues["Filters"].add_command(label='Bilateral', command=self.filters_bilateral_menu_bare_command) # 1 -> 1
        menues["Filters"].add_command(label='Median', command=self.filters_median_menu_bare_command) # 1 -> 1
        menues["Filters"].add_command(label='Lablacian', command=self.filters_lablacian_menu_bare_command) # 1 -> 1

        menues["Histogram based operations"].add_command(label='Translation', command=self.histogram_based_operations_translation_menu_bare_command) # 1 -> 1
        menues["Histogram based operations"].add_command(label='Inverse', command=self.histograme_based_operations_inverse_menu_bare_command) # 1 -> 1
        menues["Histogram based operations"].add_command(label='Dynamic expansion', command=self.histograme_based_operations_dynamic_expansion_menu_bare_command) # 1 -> 1
        menues["Histogram based operations"].add_command(label='Equalization', command=self.histogram_based_operations_equalization_menu_bare_command) # 1 -> 1
        menues["Histogram based operations"].add_command(label='Histogram matching', command=self.histogram_based_operations_histogram_matching_menu_bare_command) # 2 -> 1 

        menues["Segmentation"].add_command(label='Object detection', command=self.segmentation_object_detection_menu_bare_command) # 1 -> n 
        menues["Segmentation"].add_command(label='Edge detection', command=self.segmentation_edge_detection_menu_bare_command) # 1 -> 1

        menues["Visualization"].add_command(label='Show Image', command=self.visualization_show_menu_bare_command) # 1 -> 0
        menues["Visualization"].add_command(label='Show Histogram', command=self.visualization_show_show_histograms) # 1 -> 0
        menues["Visualization"].add_command(label='Show Metadata', command=self.visualization_show_meta_data_menu_bare_command) # 1 -> 0
        """ MEAN , STANDAR DEVIATION ,MEDIAN , Outliers, DIMENSIONS , IMAGE MODE RGB,L"""

    def __create_operations_stack__(self):
        self.operation_stack_tree_view = ttk.Treeview(master=self, columns=('N°', 'Operation', 'Args'), show='headings',selectmode="extended")

        self.operation_stack_tree_view.heading('N°', text='N°')
        self.operation_stack_tree_view.heading('Operation', text='Operation')
        self.operation_stack_tree_view.heading('Args', text='Args')

        self.operation_stack_tree_view.pack(expand=True, fill=tk.BOTH)
    
    def __create_buttons__(self):
        self.dlt_btn = ttk.Button(self,text="Discard",command=self.btn_dlt_command)
        self.cpy_btn = ttk.Button(self,text="Copy to top",command=self.btn_cpy_command)
        self.clr_btn = ttk.Button(self,text="Clear",command=self.btn_clr_command)

        self.clr_btn.pack(fill=tk.BOTH,padx=5,pady=3)
        self.cpy_btn.pack(fill=tk.BOTH,padx=5,pady=3) 
        self.dlt_btn.pack(fill=tk.BOTH,padx=5,pady=3)

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
    
    def geometric_operations_reflection_menu_bare_command(self):
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
        direction  = simpledialog.askstring("REFLECTION",'H FOR HORIZANTALL AND V FOR VERTICAL')
        if direction is None:
            messagebox.showerror("ERROR","OPERATION WAS CANCELED")
            return
        direction = direction.lower().strip()
        if direction not in ['h','v']:
            messagebox.showerror("ERROR",f"DIRECTION MUST BE H OR V BUT {direction} WAS GIVEN")
            return
        
        self.operation_stack.append(StackFrame(input_imgs,ift.MyImage.reflecte,(direction,),'o'))
        self.redraw_operation_stack_tree_view()

    def geometric_operations_rescale_menu_bare_command(self):
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

        x = simpledialog.askfloat("Re-SCALE FACTORS","ENTER THE RESCALING FACTOR FOR THE X AXE")
        if x is None:
            messagebox.showerror("ERROR","OPERATION WAS CANCELED")
            return
        y = simpledialog.askfloat("Re-SCALE FACTORS","ENTER THE RESCALING FACTOR FOR THE Y AXE")
        if y is None:
            messagebox.showerror("ERROR","OPERATION WAS CANCELED")
            return
        
        if y <= 0 or x <= 0:
            messagebox.showerror("ERROR","X FACTOR AND Y FACTOR MUST BE POSITIVE")
            return
        
        self.operation_stack.append(StackFrame(input_imgs,ift.MyImage.rescale,(x,y),'o'))
        self.redraw_operation_stack_tree_view()
    
    def geometric_operatioins_cut_menu_bare_command(self):
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

        x = simpledialog.askinteger("X POSITION","ENTER THE X POSITION FOR THE UPPER LEFT CORNER OF THE IMAGE")
        if x is None:
            messagebox.showerror("ERROR","OPERATION WAS CANCELED")
            return
        if not 0<=x<input_imgs.width:
            messagebox.showerror("ERROR","X IS OUT OF BOUND")
            return
        
        y = simpledialog.askinteger("Y POSITION","ENTER THE Y POSITION FOR THE UPPER LEFT CORNER OF THE IMAGE")
        if y is None:
            messagebox.showerror("ERROR","OPERATION WAS CANCELED")
            return
        if not 0<=y<input_imgs.height:
            messagebox.showerror("ERROR","Y IS OUT OF BOUND")
            return
        
        w = simpledialog.askinteger("WIDTH","ENTER THE WIDTH OF THE SUB-IMAGE")
        if w is None:
            messagebox.showerror("ERROR","OPERATION WAS CANCELED")
            return
        if w <= 0:
            messagebox.showerror("ERROR","THE WIDTH MUST BE A POSTIVE INTEGER")
            return
        h = simpledialog.askinteger("HEIGHT","ENTER THE HEIGHT OF THE SUB-IMAGE")
        if h is None:
            messagebox.showerror("ERROR","OPERATION WAS CANCELED")
            return
        if h <= 0:
            messagebox.showerror("ERROR","THE HEIGHT MUST BE A POSTIVE INTEGER")
            return

        self.operation_stack.append(StackFrame(input_imgs,ift.MyImage.cut,(x,y,w,h),'o'))
        self.redraw_operation_stack_tree_view()

    def geometric_operatioins_past_on_canvas_menu_bare_command(self):
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

        # W
        w = simpledialog.askinteger("WIDTH","ENTER THE WIDTH OF THE CANVAS")
        if w is None:
            messagebox.showerror("ERROR","OPERATION WAS CANCELED")
            return
        if w <= 0:
            messagebox.showerror("ERROR","THE WIDTH MUST BE A POSTIVE INTEGER")
            return
        if w < input_imgs.width:
            messagebox.showerror('ERROR',"THE WIDTH OF THE CANVAS MUST BE LARGER THAN THE IMAGE")
            return
        # H
        h = simpledialog.askinteger("HEIGHT","ENTER THE HEIGHT OF THE CANVAS")
        if h is None:
            messagebox.showerror("ERROR","OPERATION WAS CANCELED")
            return
        if h <= 0:
            messagebox.showerror("ERROR","THE HEIGHT MUST BE A POSTIVE INTEGER")
            return
        if h < input_imgs.height:
            messagebox.showerror('ERROR',"THE HEIGHT OF THE CANVAS MUST BE LARGER THAT THE IMAGE")
            return

        x = simpledialog.askinteger("X POSITION","ENTER THE X POSITION FOR THE UPPER LEFT CORNER WHERE TO PASTE THE IMAGE")
        if x is None:
            messagebox.showerror("ERROR","OPERATION WAS CANCELED")
            return
        if not 0<=x<w:
            messagebox.showerror("ERROR","X IS OUT OF THE BOUNDS OF THE CANVAS")
            return
        
        y = simpledialog.askinteger("Y POSITION","ENTER THE Y POSITION FOR THE UPPER LEFT CORNER WHERE TO PASTE THE IMAGE")
        if y is None:
            messagebox.showerror("ERROR","OPERATION WAS CANCELED")
            return
        if not 0<=y<h:
            messagebox.showerror("ERROR","Y IS OUT OF THE BOUNDS OF THE CANVAS")
            return
        
        self.operation_stack.append(StackFrame(input_imgs,ift.MyImage.paste,(x,y,w,h),'o'))
        self.redraw_operation_stack_tree_view()
    
    def geometric_operatioins_overlay_menu_bare_command(self):
        if len(self.operation_stack) == 0:
            messagebox.showerror("ERROR","NO IMAGE WAS PROVIDED")
            return
        images = self.get_selected_images()
        if len(images) == 0:
            messagebox.showerror("ERROR","NO IMAGE WAS SELECTED")
            return
        if len(images) == 1:
            messagebox.showerror('ERROR',"CAN'T OVERLAY ONE IMAGE")
            return
        W = max([img.width for img in images])
        H = max([img.height for img in images])

        result = images[0]
        for img in images[1:]:
            if img.width != W or img.height != H:
                self.operation_stack.append(StackFrame(img,ift.MyImage.rescale,(W/img.width,H/img.height),'o'))
                self.operation_stack.append(StackFrame(result,ift.MyImage.lay,(self.operation_stack[-1].imgs_out[0],),'o')) 
                result = self.operation_stack[-1].imgs_out[0]
            else:
                self.operation_stack.append(StackFrame(result,ift.MyImage.lay,(img,),'o')) 

        self.redraw_operation_stack_tree_view()

    # PHOTOMETRIC OPERATIOSN
    def photometric_operation_gray_scale_menu_bare_command(self):
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

        self.operation_stack.append(StackFrame(input_imgs,ift.MyImage.gray_scale,(),'o'))
        self.redraw_operation_stack_tree_view()
    
    def photometric_operation_resolution_under_scaling(self):
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

        factor = simpledialog.askinteger("FACTOR","ENTER THE FACTOR BETWEEN 2<= factor <256,NOTE : FACTOR MUST divide 256")
        if factor is None:
            messagebox.showerror("ERROR","OPERATION WAS CANCELED")
            return
        if not (2<=factor<256):
            messagebox.showerror('ERROR',"THE FACTOR MUST BE BETWEEN 2<= factor <256")
            return
        if 256 % factor != 0:
            messagebox.showerror('ERROR',"FACTOR MUST divide 256")
            return
        self.operation_stack.append(StackFrame(input_imgs,ift.MyImage.resolution_underscaling,(factor,),'o'))
        self.redraw_operation_stack_tree_view()

    # Histogram based operations
    def histogram_based_operations_translation_menu_bare_command(self):
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

        shift = simpledialog.askinteger("SHIFT AMOUNT","ENTER THE SHIFT VALUE")
        if shift is None:
            messagebox.showerror("ERROR","OPERATION WAS CANCELED")
            return
        
        self.operation_stack.append(StackFrame(input_imgs,ift.MyImage.histo_translation,(shift,),'o'))
        self.redraw_operation_stack_tree_view()

    def histograme_based_operations_inverse_menu_bare_command(self):
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
        
        self.operation_stack.append(StackFrame(input_imgs,ift.MyImage.histo_inverse,(),'o'))
        self.redraw_operation_stack_tree_view()
    
    def histograme_based_operations_dynamic_expansion_menu_bare_command(self):
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

        self.operation_stack.append(StackFrame(input_imgs,ift.MyImage.histo_expansion_dynamique,(),'o'))
        self.redraw_operation_stack_tree_view()
    
    def histogram_based_operations_equalization_menu_bare_command(self):
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

        self.operation_stack.append(StackFrame(input_imgs,ift.MyImage.histo_equalisation,(),'o'))
        self.redraw_operation_stack_tree_view()

    def histogram_based_operations_histogram_matching_menu_bare_command(self):
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

        # selecting the modle
        path = filedialog.askopenfilename()
        if path is None or path == '':
            messagebox.showerror('ERROR',"THE OPERATION WAS CANCELLED")
            return
        path = path.strip()

        if not any([path.endswith(ext) for ext in ['.jpeg',".png",".jpg"]]):
            x = ['.jpeg',".png",".jpg"]
            messagebox.showerror("ERROR",f"CHOOSE AN IMAGE OF TYPE {x}")
            return
        mdl = ift.MyImage.open_image_as_rgb_matrices(path)
        if input_imgs.mode == "L":
            mdl = mdl.gray_scale()
        
        self.operation_stack.append(StackFrame(input_imgs,ift.MyImage.histo_matching,(mdl,),'o'))
        self.redraw_operation_stack_tree_view()

    # Filter
    def filters_mean_menu_bare_command(self):
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

        size = simpledialog.askinteger("MEAN FILTER SIZE","ENTER THE SIZE OF THE MEAN FILTER KERNEL")
        if size is None:
            messagebox.showerror("ERROR","THE OPERATION WAS CANCELLED")
            return
        if size <= 0:
            messagebox.showerror('ERROR',"SIZE OF THE MEAN FILTER KERNEL MUST BE POSITIVE")
            return
        if size % 2 == 0:
            messagebox.showerror('ERROR',"MEAN FILTER KERNEL MUST HAVE AN ODD SIZE")
            return
        
        if size > input_imgs.width * input_imgs.height:
            messagebox.showerror('ERROR',"THE KERNEL SIZE IS BEGGER THAN THE IMAGE")
            return
        
        self.operation_stack.append(StackFrame(input_imgs,ift.MyImage.mean_filter,(size,),'o'))
        self.redraw_operation_stack_tree_view()

    def filters_gaussian_menu_bare_command(self):
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

        size = simpledialog.askinteger("GAUSSIAN FILTER SIZE","ENTER THE SIZE OF THE GAUSSIAN FILTER KERNEL")
        if size is None:
            messagebox.showerror("ERROR","THE OPERATION WAS CANCELLED")
            return
        if size <= 0:
            messagebox.showerror('ERROR',"SIZE OF THE GAUSSIAN FILTER KERNEL MUST BE POSITIVE")
            return
        if size % 2 == 0:
            messagebox.showerror('ERROR',"GAUSSIAN FILTER KERNEL MUST HAVE AN ODD SIZE")
            return
        
        if size > input_imgs.width * input_imgs.height:
            messagebox.showerror('ERROR',"THE KERNEL SIZE IS BEGGER THAN THE IMAGE")
            return
        
        std = simpledialog.askfloat("STANDARE DEVIATION","ENTER THE VALUE OF THE STANDAR DEVIATIONS")
        if std is None:
            messagebox.showerror('ERROR',"OPERATION WAS CANCELLED")
            return
        if std <= 0:
            messagebox.showerror("ERROR","STANDAR DEVIATION MUST BE POSITIVE")
            return
        

        self.operation_stack.append(StackFrame(input_imgs,ift.MyImage.gaussian_filter,(size,std),'o'))
        self.redraw_operation_stack_tree_view()

    def filters_median_menu_bare_command(self):
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

        size = simpledialog.askinteger("MEDIAN FILTER SIZE","ENTER THE SIZE OF THE MEDIAN FILTER KERNEL")
        if size is None:
            messagebox.showerror("ERROR","THE OPERATION WAS CANCELLED")
            return
        if size <= 0:
            messagebox.showerror('ERROR',"SIZE OF THE MEDIAN FILTER KERNEL MUST BE POSITIVE")
            return
        if size % 2 == 0:
            messagebox.showerror('ERROR',"MEDIAN FILTER KERNEL MUST HAVE AN ODD SIZE")
            return
        
        if size > input_imgs.width * input_imgs.height:
            messagebox.showerror('ERROR',"THE KERNEL SIZE IS BEGGER THAN THE IMAGE")
            return
        
        self.operation_stack.append(StackFrame(input_imgs,ift.MyImage.median_filter,(size,),'o'))
        self.redraw_operation_stack_tree_view()

    def filters_lablacian_menu_bare_command(self):
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

        size = simpledialog.askinteger("LAPLACIAN FILTER SIZE","ENTER THE SIZE OF THE LAPLACIAN FILTER KERNEL")
        if size is None:
            messagebox.showerror("ERROR","THE OPERATION WAS CANCELLED")
            return
        if size <= 0:
            messagebox.showerror('ERROR',"SIZE OF THE LAPLACIAN FILTER KERNEL MUST BE POSITIVE")
            return
        if size % 2 == 0:
            messagebox.showerror('ERROR',"LAPLACIAN FILTER KERNEL MUST HAVE AN ODD SIZE")
            return
        
        if size > input_imgs.width * input_imgs.height:
            messagebox.showerror('ERROR',"THE KERNEL SIZE IS BEGGER THAN THE IMAGE")
            return
        
        distance = simpledialog.askinteger("DISTANCE TYPE","CHOOSE THE DISTANCE FOR THE LAPLACIAN FILTER 1 FOR MANHATTEN AND 2 FOR MAX")
        if distance is None:
            messagebox.showerror("ERROR","THE OPERATION WAS CANCELLED")
            return
        if distance not in (1,2):
            messagebox.showerror("ERROR","UNSOPPRTED DISTANCE")
            return
        distance = 'max' if distance == 2 else "manhatten"
        self.operation_stack.append(StackFrame(input_imgs,ift.MyImage.laplacian_sharpning_filter,(distance,size),'o'))
        self.redraw_operation_stack_tree_view()
    
    def filters_bilateral_menu_bare_command(self):
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

        size = simpledialog.askinteger("BILETARAL FILTER SIZE","ENTER THE SIZE OF THE BILETARAL FILTER KERNEL")
        if size is None:
            messagebox.showerror("ERROR","THE OPERATION WAS CANCELLED")
            return
        if size <= 0:
            messagebox.showerror('ERROR',"SIZE OF THE BILETARAL FILTER KERNEL MUST BE POSITIVE")
            return
        if size % 2 == 0:
            messagebox.showerror('ERROR',"BILETARAL FILTER KERNEL MUST HAVE AN ODD SIZE")
            return
        if size > input_imgs.width * input_imgs.height:
            messagebox.showerror('ERROR',"THE KERNEL SIZE IS BEGGER THAN THE IMAGE")
            return
        std_s = simpledialog.askfloat("STANDARE DEVIATION","ENTER THE VALUE OF THE STANDAR DEVIATIONS FOR THE SPATIAL GAUSSIAN")
        if std_s is None:
            messagebox.showerror('ERROR',"OPERATION WAS CANCELLED")
            return
        if std_s <= 0:
            messagebox.showerror("ERROR","STANDAR DEVIATION MUST BE POSITIVE")
            return
        
        std_b = simpledialog.askfloat("STANDARE DEVIATION","ENTER THE VALUE OF THE STANDAR DEVIATIONS FRO THE BRIGHTNESS GAUSSIAN")
        if std_b is None:
            messagebox.showerror('ERROR',"OPERATION WAS CANCELLED")
            return
        if std_b <= 0:
            messagebox.showerror("ERROR","STANDAR DEVIATION MUST BE POSITIVE")
            return
        
        self.operation_stack.append(StackFrame(input_imgs,ift.MyImage.bilateral_filter,(size,std_s,std_b),'o'))
        self.redraw_operation_stack_tree_view()
    
    # Segmentation
    def segmentation_edge_detection_menu_bare_command(self):
        if len(self.operation_stack) == 0:
            messagebox.showerror("ERROR", "NO IMAGE WAS PROVIDED")
            return

        input_imgs = self.operation_stack[-1].imgs_out
        if len(input_imgs) <= 0:
            messagebox.showerror("ERROR", "ERROR NO IMAGE WAS FOUND THIS ERROR SHOUD NEVER HAPPEN CONTACT DEVS")
            return

        if len(input_imgs) > 1:
            messagebox.showerror("ERROR",
                                "THE LAST OPERATION GENERATED MORE THAN ONE IMAGES PLEASE COMBINE THEM USING OVERLAYING OR DISCARD THEM")
            return
        input_imgs = input_imgs[0]

        threshold = simpledialog.askinteger("THRESHOLD", "ENTER THE VALUE OF THRESHOLD BETWEEN 1 and 255")
        if threshold is None:
            messagebox.showerror("ERROR", "THE OPERATION WAS CANCELLED")
            return
        if threshold <= 0:
            messagebox.showerror('ERROR', "THRESHOLD OF THE EDGE DETECTION MUST BE POSITIVE")
            return

        if threshold > 255:
            messagebox.showerror('ERROR', "THE TRESHOLD IS BIGGER THAN 255")
            return

        operation = simpledialog.askinteger("OPERATOR TYPE",
                                            "CHOOSE THE OPERATOR FOR THE EDGE DETECTION\n1 FOR ROBERT\n2 FOR SOBEL\n3 FOR PREWITT")
        if operation is None:
            messagebox.showerror("ERROR", "THE OPERATION WAS CANCELLED")
            return
        if operation not in (1, 2, 3):
            messagebox.showerror("ERROR", "UNSOPPRTED OPTION")
            return
        
        if operation == 1:
            self.operation_stack.append(
                StackFrame(input_imgs, ift.MyImage.edge_detection_robert, (threshold,), 'o'))
            
        elif operation == 2:
            self.operation_stack.append(
                StackFrame(input_imgs, ift.MyImage.edge_detection_sobel, (threshold,), 'o'))
        elif operation == 3:
            self.operation_stack.append(
                StackFrame(input_imgs, ift.MyImage.edges_detection_prewitt, (threshold,), 'o'))
        else:
            messagebox.showerror("ERROR",f"THERE IS NO OPERATOR FOR THE SELECTED OPTION {operation}")
            return

        self.redraw_operation_stack_tree_view()

    def segmentation_object_detection_menu_bare_command(self):
        if len(self.operation_stack) == 0:
            messagebox.showerror("ERROR", "NO IMAGE WAS PROVIDED")
            return

        input_imgs = self.operation_stack[-1].imgs_out
        if len(input_imgs) <= 0:
            messagebox.showerror("ERROR", "ERROR NO IMAGE WAS FOUND THIS ERROR SHOUD NEVER HAPPEN CONTACT DEVS")
            return

        if len(input_imgs) > 1:
            messagebox.showerror("ERROR",
                                "THE LAST OPERATION GENERATED MORE THAN ONE IMAGES PLEASE COMBINE THEM USING OVERLAYING OR DISCARD THEM")
            return
        input_imgs = input_imgs[0]

        k = simpledialog.askinteger("K VALUE","ENTER THE NUMBER OF CLUSTERS")
        if k is None:
            return
        if k<=0:
            messagebox.showerror('ERROR',"THE NUMBER OF CLUSTERS MUST BE POSITIVE")
            return
        if k > 20:
            messagebox.showerror('ERROR',"THE MAXIMUM SUPPORTED FOR K IS 20")
            return
        use_tagging = messagebox.askyesno("TAGGING","WHOULD YOU LIKE TO USE THE BINARY TAGGING ON EACH CLUSTER ?")
        
        self.operation_stack.append(StackFrame(input_imgs,ift.MyImage.kmean,(k,),'o'))
        if use_tagging:
            kmean_clusters = self.operation_stack[-1].imgs_out 
            for img in kmean_clusters:
                self.operation_stack.append(StackFrame(img,ift.MyImage.binary_tagging,(),'o'))
        
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

    def visualization_show_meta_data_menu_bare_command(self):
        if len(self.operation_stack) == 0:
            messagebox.showerror("ERROR","NO IMAGE WAS PROVIDED")
            return
        
        imgs =  self.get_selected_images()
        if len(imgs) <= 0:
            messagebox.showerror("ERROR","ERROR NO IMAGE WAS SELCTED")
            return
        
        if self.meta_data_frame is not None:
            self.meta_data_frame.destroy()
        
        self.meta_data_frame = MetaDataFrame(self,imgs)
        self.meta_data_frame.run()

    def visualization_show_show_histograms(self):
        if len(self.operation_stack) == 0:
            messagebox.showerror("ERROR","NO IMAGE WAS PROVIDED")
            return
        
        imgs =  self.get_selected_images()
        if len(imgs) <= 0:
            messagebox.showerror("ERROR","ERROR NO IMAGE WAS SELCTED")
            return
        
        h = simpledialog.askinteger('HISTOGRAM TYPE',"CHOOSE THE TYPE OF THE HISTOGRAM\n1- FREQUENCY HISTOGRAM\n2-NORMILIZED HISTOGRAM\n3-CUMULATED HISTOGRAM\n4-CUMULATED NORMALIZED HISTOGRAM")
        if h is None:
            messagebox.showerror('ERROR',"OPERATION WAS CANCALLED")
            return
        if h not in (1,2,3,4):
            messagebox.showerror('ERROR',"UNCORRECT OPTION")
            return
        TYPE = {
            1:'h',
            2:'nh',
            3:'ch',
            4:"cnh",
        }
        ift.MyImage.show_histograms(imgs,TYPE[h])

    # Button
    def btn_dlt_command(self):
        selected_images  = self.get_selected_images_by_index()
        new_operation_stack = []
        for i,sframe in enumerate(self.operation_stack):
            if i not in selected_images:
                new_operation_stack.append(sframe)
        
        self.operation_stack = new_operation_stack
        self.redraw_operation_stack_tree_view()
    
    def btn_cpy_command(self):
        selected_images  = self.get_selected_images_by_index()
        new_operation_stack = []
        for i,sframe in enumerate(self.operation_stack):
            if i in selected_images:
                new_operation_stack.append(sframe)
        self.operation_stack.extend(new_operation_stack)
        self.redraw_operation_stack_tree_view()

    def btn_clr_command(self):
        self.operation_stack.clear()
        self.redraw_operation_stack_tree_view()

    # private methods
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
    
    def get_selected_images_by_index(self) -> list[int]:
        imges = []
        selected_stack_frames = self.operation_stack_tree_view.selection()
        for i in selected_stack_frames:
            imges.append(int(self.operation_stack_tree_view.item(i)["values"][0]))
        return imges


if __name__ == "__main__":
    Application().run()