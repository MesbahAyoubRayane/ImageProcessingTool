import tkinter as tk
from tkinter import ttk
from typing import Callable
import ttkbootstrap
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ttkbootstrap.window import Window
import images_function_tools as ift
from tkinter import filedialog, messagebox, simpledialog
import os
import sys
from copy import deepcopy
import numpy as np

class StackFrame:
    """
    This class will be used to accumulate all the operations on an image
    each stack frae represent an operation
    """

    def __init__(self, img: ift.MyImage | None, f: Callable, args: tuple, _type: str) -> None:
        """
        _type : must be s or o (s for static and o for objetc) 
        """
        _type = _type.lower()
        if _type not in ('s', 'o'): raise ValueError(f'the selected type {_type} is not correct')

        self.function = f
        self.img_in = img
        self.args = args
        self._type = _type

        if self._type == 'o':
            r = self.function(self.img_in, *self.args)

        elif self._type == 's':
            r = self.function(*self.args)

        else:
            raise Exception(f"type {_type} is not supported")  # i know that this clause is impossible to happen

        if not isinstance(r, list) and not isinstance(r, ift.MyImage):
            raise Exception("The only types accapted as output for the stack frame are list of MyImage or MyImage")

        if isinstance(r, ift.MyImage): r = [r]
        self.imgs_out: list[ift.MyImage] = r
        self.name = ""

    def setName(self, name: str):
        self.name = name

    def copy(self):
        return deepcopy(self)


class MetaDataFrame(tk.Toplevel):
    def __init__(self, master: tk.Frame | tk.Tk, imgs: list[ift.MyImage]):
        super().__init__(master=master)
        for i, img in enumerate(imgs):
            frm = ttk.Frame(self)

            ttk.Label(frm, text="IMAGE N°: " + str(i)).pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            ttk.Label(frm, text="DIMENSIONS : " + str(img.dimensions)).pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            ttk.Label(frm, text="IMAGE MODE : " + str(img.mode)).pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            mean = img.mean()
            if isinstance(mean, tuple):
                mean = round(mean[0], 2), round(mean[1], 2), round(mean[2], 2)
            else:
                mean = round(mean, 2)
            ttk.Label(frm, text="IMAGE MEAN : " + str(mean)).pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            median = img.median()
            if isinstance(median, tuple):
                median = round(median[0], 2), round(median[1], 2), round(median[2], 2)
            else:
                median = round(median, 2)
            ttk.Label(frm, text="IMAGE MEDIAN : " + str(median)).pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            std = img.std()
            if isinstance(std, tuple):
                std = round(std[0], 2), round(std[1], 2), round(std[2], 2)
            else:
                std = round(std, 2)
            ttk.Label(frm, text="IMAGE STANDARE DEVIATION : " + str(std)).pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            out = img.outliers()
            if isinstance(out, tuple):
                out = sum(out) / (img.width * img.height * 3)
            elif isinstance(out, int):
                out = out / (img.width * img.height)
            out = round(out * 100, 2)
            ttk.Label(frm, text="IMAGE OUTLIERS : " + str(out) + "%").pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            frm.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.resizable(False, False)

    def run(self):
        self.mainloop()


class Application(Window):
    """
    The application will be initialized by the init , inside the innit several calls for different init function
    will occur , to create the components the layout and other stuff at the end a binding function will be called to
    define behaviours
    """
    MAX_CLUSTERS_SIZE = 20
    MAX_IMG_WIDTH = 5000
    MAX_IMG_HEIGHT = 5000
    OK_IMG_TYPES = '.jpeg', ".png", ".jpg"

    def __init__(self):
        super().__init__()  # themename="superhero"
        self.title("Image processing software")

        # centering the window
        w, h = 900, 500
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x, y = (screen_width - w) // 2, (screen_height - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.minsize(w, h)
        self.__create_components__()

        # global variables
        self.meta_data_frame: MetaDataFrame | None = None
        self.operation_stack: list[StackFrame] = []

    def __create_components__(self):
        self.__create__menu_bare__()
        self.__create_operations_stack__()
        self.__create_buttons__()


    def __create__menu_bare__(self):
        menu = tk.Menu(self)
        menues: dict[str, tk.Menu] = {}
        self.configure(menu=menu)
        categories = ["Files", "Geometric operations", "Photometric operation", "Filters", "Histogram based operations",
                      "Segmentation", "Visualization"]

        for m in categories:
            sub_menu = tk.Menu(menu, tearoff=False)
            menu.add_cascade(label=m, menu=sub_menu)
            menues[m] = sub_menu

        # the Files sub menu
        menues["Files"].add_command(label='Open Image', command=self.open_image_menu_bare_command)  # 0 -> 1
        menues["Files"].add_command(label='Save Image', command=self.save_image_menu_bare_command)  # 1 -> 0
        menues["Files"].add_separator()
        menues["Files"].add_command(label='Exit', command=self.exit_menu_bare_command)

        menues["Geometric operations"].add_command(label='Translation',
                                                   command=self.geometric_operations_translation_menu_bare_command)  # 1 -> 1
        menues["Geometric operations"].add_command(label='Rotation',
                                                   command=self.geometric_operations_rotation_menu_bare_command)  # 1 -> 1
        menues["Geometric operations"].add_command(label='Reflection',
                                                   command=self.geometric_operations_reflection_menu_bare_command)  # 1 -> 1
        menues["Geometric operations"].add_command(label='Re-scale',
                                                   command=self.geometric_operations_rescale_menu_bare_command)  # 1 -> 1
        menues["Geometric operations"].add_separator()
        menues["Geometric operations"].add_command(label='Cut',
                                                   command=self.geometric_operations_cut_menu_bare_command)  # 1 -> 1
        menues["Geometric operations"].add_command(label='Past',
                                                   command=self.geometric_operations_past_on_canvas_menu_bare_command)  # 1 -> 1
        menues["Geometric operations"].add_separator()
        menues["Geometric operations"].add_command(label='Overlay',
                                                   command=self.geometric_operatoins_overlay_menu_bare_command)  # n -> 1

        menues["Photometric operation"].add_command(label='Gray scale',
                                                    command=self.photometric_operation_gray_scale_menu_bare_command)  # 1 -> 1
        menues["Photometric operation"].add_command(label='Resolution under-scaling',
                                                    command=self.photometric_operation_resolution_under_scaling)  # 1 -> 1

        menues["Filters"].add_command(label='Mean', command=self.filters_mean_menu_bare_command)  # 1 -> 1
        menues["Filters"].add_command(label='Median', command=self.filters_median_menu_bare_command)  # 1 -> 1
        menues["Filters"].add_command(label='Max', command=self.filters_max_menu_bare_command)  # 1 -> 1
        menues["Filters"].add_command(label='Min', command=self.filters_min_menu_bare_command)  # 1 -> 1
        menues["Filters"].add_separator()
        menues["Filters"].add_command(label='Gaussian', command=self.filters_gaussian_menu_bare_command)  # 1 -> 1
        menues["Filters"].add_command(label='Bilateral', command=self.filters_bilateral_menu_bare_command)  # 1 -> 1
        menues["Filters"].add_command(label='Laplacian', command=self.filters_lablacian_menu_bare_command)  # 1 -> 1

        menues["Histogram based operations"].add_command(label='Translation',
                                                         command=self.histogram_based_operations_translation_menu_bare_command)  # 1 -> 1
        menues["Histogram based operations"].add_command(label='Inverse',
                                                         command=self.histogram_based_operations_inverse_menu_bare_command)  # 1 -> 1
        menues["Histogram based operations"].add_separator()
        menues["Histogram based operations"].add_command(label='Dynamic expansion',
                                                         command=self.histogram_based_operations_dynamic_expansion_menu_bare_command)  # 1 -> 1
        menues["Histogram based operations"].add_command(label='Equalization',
                                                         command=self.histogram_based_operations_equalization_menu_bare_command)  # 1 -> 1
        menues["Histogram based operations"].add_command(label='Histogram matching',
                                                         command=self.histogram_based_operations_histogram_matching_menu_bare_command)  # 2 -> 1

        menues["Segmentation"].add_command(label='Object detection (Kmean)',
                                           command=self.segmentation_object_detection_kmean_menu_bare_command)  # 1 -> n
        menues["Segmentation"].add_command(label='Object detection (Threshold)',
                                           command=self.segmentation_object_detection_threshold_menu_bare_command)
        menues["Segmentation"].add_command(label='Binary Tagging',
                                           command=self.segmentation_binary_tagging_menu_bare_command)
        menues["Segmentation"].add_separator()
        menues["Segmentation"].add_command(label='Edge detection',
                                           command=self.segmentation_edge_detection_menu_bare_command)  # 1 -> 1

        menues["Visualization"].add_command(label='Show Image',
                                            command=self.visualization_show_menu_bare_command)  # 1 -> 0
        menues["Visualization"].add_command(label='Show Histogram',
                                            command=self.visualization_show_show_histograms)  # 1 -> 0
        menues["Visualization"].add_command(label='Show Metadata',
                                            command=self.visualization_show_meta_data_menu_bare_command)  # 1 -> 0

    def __create_operations_stack__(self):
        self.operation_stack_tree_view = ttk.Treeview(master=self, columns=('N°', 'Operation', 'Args', "Name"),
                                                      show='headings', selectmode="extended")

        self.operation_stack_tree_view.heading('N°', text='N°', anchor=tk.CENTER)
        self.operation_stack_tree_view.heading('Operation', text='Operation', anchor=tk.CENTER)
        self.operation_stack_tree_view.heading('Args', text='Args', anchor=tk.CENTER)
        self.operation_stack_tree_view.heading('Name', text="Name", anchor=tk.CENTER)

        self.operation_stack_tree_view.pack(expand=True, fill=tk.BOTH)

        # BINDING BEHAVIOURS
        def dblclk_show_img(_):
            if len(self.get_selected_images()) == 0: return
            self.visualization_show_menu_bare_command()

        def dblclk_show_meta(_):
            if len(self.get_selected_images()) == 0: return
            self.visualization_show_meta_data_menu_bare_command()

        self.operation_stack_tree_view.bind("<Double-1>", dblclk_show_img)
        self.operation_stack_tree_view.bind("<Double-3>", dblclk_show_meta)

    def __create_buttons__(self):
        self.dlt_btn = ttk.Button(self, text="Discard", command=self.btn_dlt_command, bootstyle=ttkbootstrap.DANGER)
        self.clr_btn = ttk.Button(self, text="Clear", command=self.btn_clr_command, bootstyle=ttkbootstrap.DANGER)
        self.cpy_btn = ttk.Button(self, text="Copy", command=self.btn_cpy_command, bootstyle=ttkbootstrap.PRIMARY)
        self.extract_btn = ttk.Button(self, text="Extract", command=self.btn_extract_command,
                                      bootstyle=ttkbootstrap.PRIMARY)
        self.rename_btn = ttk.Button(self, text="Rename", command=self.btn_rename_command,
                                     bootstyle=ttkbootstrap.SUCCESS)

        self.rename_btn.pack(fill=tk.BOTH, padx=5, pady=3)
        self.cpy_btn.pack(fill=tk.BOTH, padx=5, pady=3)
        self.extract_btn.pack(fill=tk.BOTH, padx=5, pady=3)
        self.dlt_btn.pack(fill=tk.BOTH, padx=5, pady=3)
        self.clr_btn.pack(fill=tk.BOTH, padx=5, pady=3)

    def run(self):
        self.mainloop()

    # menu bare function
    def open_image_menu_bare_command(self):
        path = filedialog.askopenfilename()
        if path is None or len(path) == 0:
            messagebox.showerror("ERROR", "PROVIDE A CORRECT PATH TO THE IMAGE")
            return

        img = None
        try:
            img = ift.MyImage.open_image_as_rgb_matrices(path)
        except Exception as e:
            print(e)
            messagebox.showerror("ERROR", "UNCORRECT PATH")
            return

        self.operation_stack.append(StackFrame(None, ift.MyImage.open_image_as_rgb_matrices, (path,), 's'))

        self.redraw_operation_stack_tree_view()

    def save_image_menu_bare_command(self):
        if len(self.operation_stack) == 0:
            messagebox.showerror("ERROR", "NO IMAGE TO SAVE")
            return

        # testing the path
        path = filedialog.askdirectory()
        if path is None or not path:
            messagebox.showerror("EROOR", "PATH IS NOT CORRECT")
            return
        path = path.strip()

        # image name
        image_name = simpledialog.askstring("INPUT", "ENTER THE IMAGE NAME")
        if image_name is not None:
            image_name = image_name.strip()

        if image_name is None or image_name == "":
            messagebox.showerror('ERROR', "PROVIDE A CORRECT IMAGE NAME")
            return

        extension = ""
        for e in ".png", ".jpeg", ".jpg":
            if image_name.lower().endswith(e):
                break
        if extension == "": extension = ".png"

        imgs: list[ift.MyImage] = self.get_selected_images()
        if len(imgs) == 0: messagebox.showerror("ERROR", "NO IMAGE WASD PROVIDED")
        try:
            for i, img in enumerate(imgs):
                # if not extension was provided add png by default
                imgi_name = os.path.join(path, image_name + str(i) + extension)
                img.save_image(imgi_name)

        except Exception as e:
            messagebox.showerror("ERROR", f"CAN'T SAVE IMAGE")
            print(type(e))
            return
        x = "s" if len(imgs) > 1 else ""
        messagebox.showinfo("INFO", f"IMAGE{x} WERE SAVED")

    def exit_menu_bare_command(self):
        if messagebox.askyesno("CONFIRMATION", "WOULD YOU LIKE TO EXIT THE APPLICATION"):
            sys.exit(0)

    # GEOMETRIC OPERATION
    def geometric_operations_translation_menu_bare_command(self):
        input_imgs = self.test_images_for_unary_operations()
        if input_imgs is None:
            return

        x = simpledialog.askinteger("TRANSLATION", "ENTER NUMBER OF PIXELS FOR TRANSLATION OF THE X AXES")
        if x is None:
            messagebox.showerror("ERROR", "OPERATION WAS CANCELED")
            return

        y = simpledialog.askinteger("TRANSLATION", "ENTER NUMBER OF PXLS FOR TRANSLATION OF THE Y AXES")
        if y is None:
            messagebox.showerror("ERROR", "OPERATION WAS CANCELED")
            return

        self.operation_stack.append(StackFrame(input_imgs, ift.MyImage.translation, (x, y,), 'o'))
        self.redraw_operation_stack_tree_view()

    def geometric_operations_rotation_menu_bare_command(self):
        input_imgs = self.test_images_for_unary_operations()
        if input_imgs is None:
            return
        theta = simpledialog.askinteger("USER INPUT", "ENTER THE ANGLES OF ROTATION (IN DEGREES)")
        if theta is None:
            return

        self.operation_stack.append(StackFrame(input_imgs, ift.MyImage.rotate, (theta,), 'o'))
        self.redraw_operation_stack_tree_view()

    def geometric_operations_reflection_menu_bare_command(self):
        input_imgs = self.test_images_for_unary_operations()
        if input_imgs is None:
            return

        direction = simpledialog.askstring("USER INPUT", 'SELECT THE AXES OF REFLECTION\n- H(HORIZONTAL)\n-V(VERTICAL)')
        if direction is None:
            return
        direction = direction.lower().strip()
        if direction not in ('h', 'v'):
            messagebox.showerror("VALUE ERROR",
                                 f"DIRECTION MUST BE ON OF THESE VARIANTS (H,V) BUT {direction} WAS ENTERED")
            return

        self.operation_stack.append(StackFrame(input_imgs, ift.MyImage.reflect, (direction,), 'o'))
        self.redraw_operation_stack_tree_view()

    def geometric_operations_rescale_menu_bare_command(self):
        input_imgs = self.test_images_for_unary_operations()
        if input_imgs is None:
            return

        x = simpledialog.askfloat("Re-SCALE", "ENTER THE NEW WIDTH\n - 0 USE OLD WIDTH ")
        if x is None:
            return
        y = simpledialog.askfloat("Re-SCALE", "ENTER THE NEW HEIGHT \n - 0 USE OLD HEIGHT")
        if y is None:
            return

        if y < 0. or x < 0.:
            messagebox.showerror("ERROR", "WIDTH AND HEIGHT MUST BE POSITIVE")
            return

        if y == 0.:
            y = input_imgs.height
        if x == 0.:
            x = input_imgs.width

        if x > Application.MAX_IMG_WIDTH:
            messagebox.showerror("ERROR", "EXCEEDING THE MAX POSSIBLE WIDTH")
            return

        if y > Application.MAX_IMG_HEIGHT:
            messagebox.showerror("ERROR", "EXCEEDING THE MAX POSSIBLE HEIGHT")
            return

        x /= input_imgs.width
        y /= input_imgs.height

        self.operation_stack.append(StackFrame(input_imgs, ift.MyImage.rescale, (x, y), 'o'))
        self.redraw_operation_stack_tree_view()

    def geometric_operations_cut_menu_bare_command(self):
        input_imgs = self.test_images_for_unary_operations()
        if input_imgs is None:
            return
        x = simpledialog.askinteger("X POSITION", "ENTER THE X POSITION FOR THE UPPER LEFT CORNER OF THE IMAGE")
        if x is None:
            return
        if not (0 <= x < input_imgs.width):
            messagebox.showerror("ERROR", "X IS OUT OF BOUND")
            return

        y = simpledialog.askinteger("Y POSITION", "ENTER THE Y POSITION FOR THE UPPER LEFT CORNER OF THE IMAGE")
        if y is None:
            return
        if not (0 <= y < input_imgs.height):
            messagebox.showerror("ERROR", "Y IS OUT OF BOUND")
            return

        w = simpledialog.askinteger("WIDTH", "ENTER THE WIDTH OF THE SUB-IMAGE")
        if w is None:
            return
        if w <= 0:
            messagebox.showerror("ERROR", "THE WIDTH MUST BE A POSITIVE INTEGER")
            return
        h = simpledialog.askinteger("HEIGHT", "ENTER THE HEIGHT OF THE SUB-IMAGE")
        if h is None:
            return
        if h <= 0:
            messagebox.showerror("ERROR", "THE HEIGHT MUST BE A POSITIVE INTEGER")
            return

        self.operation_stack.append(StackFrame(input_imgs, ift.MyImage.cut, (x, y, w, h), 'o'))
        self.redraw_operation_stack_tree_view()

    def geometric_operations_past_on_canvas_menu_bare_command(self):
        input_imgs = self.test_images_for_unary_operations()
        if input_imgs is None:
            return

        w = simpledialog.askinteger("WIDTH", "ENTER THE WIDTH OF THE CANVAS")
        if w is None:
            return
        if w <= 0:
            messagebox.showerror("ERROR", "THE WIDTH MUST BE A POSITIVE INTEGER")
            return
        if w < input_imgs.width:
            messagebox.showerror('ERROR', "THE WIDTH OF THE CANVAS MUST BE LARGER THAN THE WIDTH OF THE IMAGE")
            return
        if w > Application.MAX_IMG_WIDTH:
            messagebox.showerror("ERROR", f"YOU HAVE EXCEEDED THE MAX WIDTH OF AN IMAGE {Application.MAX_IMG_WIDTH}")
            return
            # H
        h = simpledialog.askinteger("HEIGHT", "ENTER THE HEIGHT OF THE CANVAS")
        if h is None:
            return
        if h <= 0:
            messagebox.showerror("ERROR", "THE HEIGHT MUST BE A POSITIVE INTEGER")
            return
        if h < input_imgs.height:
            messagebox.showerror('ERROR', "THE HEIGHT OF THE CANVAS MUST BE LARGER THAT THE HEIGHT OF THE IMAGE")
            return
        if h > Application.MAX_IMG_HEIGHT:
            messagebox.showerror('ERROR', f"YOU HAVE EXCEEDED THE MAX HEIGHT OF AN IMAGE {Application.MAX_IMG_HEIGHT}")
            return
        x = simpledialog.askinteger("X POSITION",
                                    "ENTER THE X POSITION WHERE TO PASTE THE IMAGE\n(THE UPPER LEFT CORNER OF THE IMAGE)")
        if x is None:
            return
        if not (0 <= x < w):
            messagebox.showerror("ERROR", "X IS OUT OF THE BOUNDS OF THE CANVAS")
            return

        y = simpledialog.askinteger("Y POSITION",
                                    "ENTER THE Y POSITION WHERE TO PASTE THE IMAGE\n(THE UPPER LEFT CORNER OF THE IMAGE)")
        if y is None:
            return
        if not (0 <= y < h):
            messagebox.showerror("ERROR", "Y IS OUT OF THE BOUNDS OF THE CANVAS")
            return

        self.operation_stack.append(StackFrame(input_imgs, ift.MyImage.paste, (x, y, w, h), 'o'))
        self.redraw_operation_stack_tree_view()

    def geometric_operatoins_overlay_menu_bare_command(self):
        images = self.get_selected_images()
        if len(images) == 0:
            messagebox.showerror("ERROR", "NO IMAGE WAS SELECTED")
            return
        if len(images) == 1:
            messagebox.showerror('ERROR', "CAN'T OVERLAY ONE IMAGE")
            return

        w = max([img.width for img in images])
        h = max([img.height for img in images])

        result = images[0]
        for img in images[1:-1]:
            if img.width != w or img.height != h:
                img = img.rescale(w / img.width, h / img.height)
            result = result.lay(img)

        if images[-1].dimensions != result.dimensions:
            w, h = result.dimensions
            result = result.rescale(images[-1].width / w, images[-1].height / h)

        self.operation_stack.append(StackFrame(images[-1], ift.MyImage.lay, (result,), 'o'))
        self.redraw_operation_stack_tree_view()

    # PHOTOMETRIC OPERATIONS
    def photometric_operation_gray_scale_menu_bare_command(self):
        input_imgs = self.test_images_for_unary_operations()
        if input_imgs is None:
            return
        self.operation_stack.append(StackFrame(input_imgs, ift.MyImage.gray_scale, (), 'o'))
        self.redraw_operation_stack_tree_view()

    def photometric_operation_resolution_under_scaling(self):
        input_imgs = self.test_images_for_unary_operations()
        if input_imgs is None:
            return

        factor = simpledialog.askinteger("FACTOR",
                                         "ENTER THE FACTOR BETWEEN 1 < factor < 256\nNOTE: FACTOR MUST divide 256")
        if factor is None:
            return
        if not (2 <= factor < 256):
            messagebox.showerror('ERROR', "THE FACTOR MUST BE BETWEEN 2<= factor <256")
            return
        if 256 % factor != 0.:
            messagebox.showerror('ERROR', "FACTOR MUST divide 256")
            return
        self.operation_stack.append(StackFrame(input_imgs, ift.MyImage.resolution_under_scaling, (factor,), 'o'))
        self.redraw_operation_stack_tree_view()

    # Histogram based operations
    def histogram_based_operations_translation_menu_bare_command(self):
        input_imgs = self.test_images_for_unary_operations()
        if input_imgs is None:
            return

        shift = simpledialog.askinteger("SHIFT AMOUNT", "ENTER THE SHIFT VALUE")
        if shift is None:
            return

        self.operation_stack.append(StackFrame(input_imgs, ift.MyImage.histo_translation, (shift,), 'o'))
        self.redraw_operation_stack_tree_view()

    def histogram_based_operations_inverse_menu_bare_command(self):
        input_imgs = self.test_images_for_unary_operations()
        if input_imgs is None:
            return

        self.operation_stack.append(StackFrame(input_imgs, ift.MyImage.histo_inverse, (), 'o'))
        self.redraw_operation_stack_tree_view()

    def histogram_based_operations_dynamic_expansion_menu_bare_command(self):
        input_imgs = self.test_images_for_unary_operations()
        if input_imgs is None:
            return

        self.operation_stack.append(StackFrame(input_imgs, ift.MyImage.histo_expansion_dynamique, (), 'o'))
        self.redraw_operation_stack_tree_view()

    def histogram_based_operations_equalization_menu_bare_command(self):
        input_imgs = self.test_images_for_unary_operations()
        if input_imgs is None:
            return

        self.operation_stack.append(StackFrame(input_imgs, ift.MyImage.histo_equalisation, (), 'o'))
        self.redraw_operation_stack_tree_view()

    def histogram_based_operations_histogram_matching_menu_bare_command(self):
        input_imgs = self.test_images_for_unary_operations()
        if input_imgs is None:
            return

        # selecting the model image
        path = filedialog.askopenfilename()
        if path is None or path == '':
            return
        path = path.strip()

        if not any([path.endswith(ext) for ext in Application.OK_IMG_TYPES]):
            messagebox.showerror("ERROR", f"CHOOSE AN IMAGE OF TYPE {Application.OK_IMG_TYPES}")
            return
        try:
            mdl = ift.MyImage.open_image_as_rgb_matrices(path)
        except Exception as e:
            messagebox.showerror('ERROR', "CAN'T OPEN THE SELECTED IMAGE")
            return
        if input_imgs.mode == "L":
            mdl = mdl.gray_scale()

        self.operation_stack.append(StackFrame(input_imgs, ift.MyImage.histo_matching, (mdl,), 'o'))
        self.redraw_operation_stack_tree_view()

    # Filter
    def filters_mean_menu_bare_command(self):
        input_imgs = self.test_images_for_unary_operations()
        if input_imgs is None:
            return

        size = simpledialog.askinteger("MEAN FILTER SIZE", "ENTER THE SIZE OF THE MEAN FILTER KERNEL")
        if size is None:
            return
        if size <= 0:
            messagebox.showerror('ERROR', "SIZE OF THE MEAN FILTER KERNEL MUST BE POSITIVE")
            return
        if int(size % 2) == 0:
            messagebox.showerror('ERROR', "MEAN FILTER KERNEL MUST HAVE AN ODD SIZE")
            return

        if size > (input_imgs.width * input_imgs.height):
            messagebox.showerror('ERROR', "THE KERNEL SIZE IS BIGGER THAN THE IMAGE")
            return

        self.operation_stack.append(StackFrame(input_imgs, ift.MyImage.mean_filter, (size,), 'o'))
        self.redraw_operation_stack_tree_view()

    def filters_gaussian_menu_bare_command(self):
        input_imgs = self.test_images_for_unary_operations()
        if input_imgs is None:
            return

        size = simpledialog.askinteger("GAUSSIAN FILTER SIZE", "ENTER THE SIZE OF THE GAUSSIAN FILTER KERNEL")
        if size is None:
            return
        if size <= 0:
            messagebox.showerror('ERROR', "SIZE OF THE GAUSSIAN FILTER KERNEL MUST BE POSITIVE")
            return
        if size % 2 == 0:
            messagebox.showerror('ERROR', "GAUSSIAN FILTER KERNEL MUST HAVE AN ODD SIZE")
            return

        if size > (input_imgs.width * input_imgs.height):
            messagebox.showerror('ERROR', "THE KERNEL SIZE IS BIGGER THAN THE IMAGE")
            return

        std = simpledialog.askfloat("STANDAER DEVIATION", "ENTER THE VALUE OF THE STANDER DEVIATIONS")
        if std is None:
            return
        if std <= 0:
            messagebox.showerror("ERROR", "STANDER DEVIATION MUST BE POSITIVE")
            return

        self.operation_stack.append(StackFrame(input_imgs, ift.MyImage.gaussian_filter, (size, std), 'o'))
        self.redraw_operation_stack_tree_view()

    def filters_median_menu_bare_command(self):
        input_imgs = self.test_images_for_unary_operations()
        if input_imgs is None:
            return

        size = simpledialog.askinteger("MEDIAN FILTER SIZE", "ENTER THE SIZE OF THE MEDIAN FILTER KERNEL")
        if size is None:
            return
        if size <= 1:
            messagebox.showerror('ERROR', "SIZE OF THE MEDIAN FILTER KERNEL MUST >= 1")
            return
        if size % 2 == 0:
            messagebox.showerror('ERROR', "MEDIAN FILTER KERNEL MUST HAVE AN ODD SIZE")
            return

        if size > (input_imgs.width * input_imgs.height):
            messagebox.showerror('ERROR', "THE KERNEL SIZE IS BIGGER THAN THE IMAGE")
            return

        self.operation_stack.append(StackFrame(input_imgs, ift.MyImage.median_filter, (size,), 'o'))
        self.redraw_operation_stack_tree_view()

    def filters_max_menu_bare_command(self):
        input_imgs = self.test_images_for_unary_operations()
        if input_imgs is None:
            return

        size = simpledialog.askinteger("MAX FILTER SIZE", "ENTER THE SIZE OF THE MAX FILTER KERNEL")
        if size is None:
            return
        if size <= 0:
            messagebox.showerror('ERROR', "SIZE OF THE MAX FILTER KERNEL MUST BE POSITIVE")
            return
        if size % 2 == 0:
            messagebox.showerror('ERROR', "MAX FILTER KERNEL MUST HAVE AN ODD SIZE")
            return

        if size > (input_imgs.width * input_imgs.height):
            messagebox.showerror('ERROR', "THE KERNEL SIZE IS BIGGER THAN THE IMAGE")
            return

        self.operation_stack.append(StackFrame(input_imgs, ift.MyImage.max_filter, (size,), 'o'))
        self.redraw_operation_stack_tree_view()

    def filters_min_menu_bare_command(self):
        input_imgs = self.test_images_for_unary_operations()
        if input_imgs is None:
            return

        size = simpledialog.askinteger("MIN FILTER SIZE", "ENTER THE SIZE OF THE MIn FILTER KERNEL")
        if size is None:
            return
        if size <= 0:
            messagebox.showerror('ERROR', "SIZE OF THE MIN FILTER KERNEL MUST BE POSITIVE")
            return
        if size % 2 == 0:
            messagebox.showerror('ERROR', "MIN FILTER KERNEL MUST HAVE AN ODD SIZE")
            return

        if size > input_imgs.width * input_imgs.height:
            messagebox.showerror('ERROR', "THE KERNEL SIZE IS BIGGER THAN THE IMAGE")
            return

        self.operation_stack.append(StackFrame(input_imgs, ift.MyImage.min_filter, (size,), 'o'))
        self.redraw_operation_stack_tree_view()

    def filters_lablacian_menu_bare_command(self):
        input_imgs = self.test_images_for_unary_operations()
        if input_imgs is None:
            return

        size = simpledialog.askinteger("LAPLACIAN FILTER SIZE", "ENTER THE SIZE OF THE LAPLACIAN FILTER KERNEL")
        if size is None:
            return
        if size <= 0:
            messagebox.showerror('ERROR', "SIZE OF THE LAPLACIAN FILTER KERNEL MUST BE POSITIVE")
            return
        if size % 2 == 0:
            messagebox.showerror('ERROR', "LAPLACIAN FILTER KERNEL MUST HAVE AN ODD SIZE")
            return

        if size > input_imgs.width * input_imgs.height:
            messagebox.showerror('ERROR', "THE KERNEL SIZE IS BIGGER THAN THE IMAGE")
            return

        distance = simpledialog.askinteger("DISTANCE TYPE",
                                           "CHOOSE THE DISTANCE FOR THE LAPLACIAN FILTER \n1- MANHATTAN \n2- MAX")
        if distance is None:
            return
        if distance not in (1, 2):
            messagebox.showerror("ERROR", "UNSUPPORTED DISTANCE")
            return
        distance = 'max' if distance == 2 else "manhattan"
        self.operation_stack.append(
            StackFrame(input_imgs, ift.MyImage.laplacian_sharpning_filter, (distance, size), 'o'))
        self.redraw_operation_stack_tree_view()

    def filters_bilateral_menu_bare_command(self):
        input_imgs = self.test_images_for_unary_operations()
        if input_imgs is None:
            return

        size = simpledialog.askinteger("BILATERAL FILTER SIZE", "ENTER THE SIZE OF THE BILATERAL FILTER KERNEL")
        if size is None:
            return
        if size <= 0:
            messagebox.showerror('ERROR', "SIZE OF THE BILATERAL FILTER KERNEL MUST BE POSITIVE")
            return
        if size % 2 == 0:
            messagebox.showerror('ERROR', "BILATERAL FILTER KERNEL MUST HAVE AN ODD SIZE")
            return
        if size > input_imgs.width * input_imgs.height:
            messagebox.showerror('ERROR', "THE KERNEL SIZE IS BIGGER THAN THE IMAGE")
            return
        std_s = simpledialog.askfloat("STANDER DEVIATION",
                                      "ENTER THE VALUE OF THE STANDER DEVIATIONS FOR THE SPATIAL GAUSSIAN")
        if std_s is None:
            return
        if std_s <= 0:
            messagebox.showerror("ERROR", "STANDER DEVIATION MUST BE POSITIVE")
            return

        std_b = simpledialog.askfloat("STANDER DEVIATION",
                                      "ENTER THE VALUE OF THE STANDER DEVIATIONS FOR THE BRIGHTNESS GAUSSIAN")
        if std_b is None:
            return
        if std_b <= 0:
            messagebox.showerror("ERROR", "STANDER DEVIATION MUST BE POSITIVE")
            return

        self.operation_stack.append(StackFrame(input_imgs, ift.MyImage.bilateral_filter, (size, std_s, std_b), 'o'))
        self.redraw_operation_stack_tree_view()

    # Segmentation
    def segmentation_edge_detection_menu_bare_command(self):
        input_imgs = self.test_images_for_unary_operations()
        if input_imgs is None:
            return

        threshold = simpledialog.askinteger("THRESHOLD", "ENTER THE VALUE OF THRESHOLD BETWEEN 1 and 255")
        if threshold is None:
            return
        if threshold <= 0:
            messagebox.showerror('ERROR', "THRESHOLD OF THE EDGE DETECTION MUST BE POSITIVE")
            return

        if threshold > 255:
            messagebox.showerror('ERROR', "THE THRESHOLD IS BIGGER THAN 255")
            return

        operation = simpledialog.askinteger("OPERATOR TYPE",
                                            "CHOOSE THE OPERATOR FOR THE EDGE DETECTION\n1-ROBERT\n2-SOBEL\n3-PREWITT")
        if operation is None:
            return

        if operation == 1:
            stckfrm = StackFrame(input_imgs, ift.MyImage.edge_detection_robert, (threshold,), 'o')
        elif operation == 2:
            stckfrm = StackFrame(input_imgs, ift.MyImage.edge_detection_sobel, (threshold,), 'o')
        elif operation == 3:
            stckfrm = StackFrame(input_imgs, ift.MyImage.edges_detection_prewitt, (threshold,), 'o')
        else:
            messagebox.showerror("ERROR", f"THERE IS NO OPERATOR FOR THE SELECTED OPTION {operation}")
            return

        self.operation_stack.append(stckfrm)
        self.redraw_operation_stack_tree_view()

    def segmentation_object_detection_kmean_menu_bare_command(self):
        input_imgs = self.test_images_for_unary_operations()
        if input_imgs is None:
            return

        k = simpledialog.askinteger("K VALUE", "ENTER THE NUMBER OF CLUSTERS")
        if k is None:
            return
        if k <= 0:
            messagebox.showerror('ERROR', "THE NUMBER OF CLUSTERS MUST BE POSITIVE")
            return
        if k > Application.MAX_CLUSTERS_SIZE:
            messagebox.showerror('ERROR', f"THE MAXIMUM SUPPORTED FOR K IS {Application.MAX_CLUSTERS_SIZE}")
            return

        self.operation_stack.append(StackFrame(input_imgs, ift.MyImage.kmean, (k,), 'o'))
        self.redraw_operation_stack_tree_view()

    def segmentation_object_detection_threshold_menu_bare_command(self):
        input_imgs = self.test_images_for_unary_operations()
        if input_imgs is None:
            return
        threshold = simpledialog.askinteger("THRESHOLD VALUE", "ENTER A NUMBER BETWEEN 0 AND 255")
        if threshold is None:
            return

        if not (0 <= threshold < 256):
            messagebox.showerror('ERROR', "THRESHOLD MUST BE BETWEEN 0 and 255")
            return

        self.operation_stack.append(StackFrame(input_imgs, ift.MyImage.segmentation_by_threshold, (threshold,), 'o'))
        self.redraw_operation_stack_tree_view()

    def segmentation_binary_tagging_menu_bare_command(self):
        input_imgs = self.test_images_for_unary_operations()
        if input_imgs is None:
            return
        seperated = messagebox.askyesno("OPTIONS", "WOULD YOU LIKE TO SEPARATE THE OBJECTS")
        self.operation_stack.append(StackFrame(input_imgs, ift.MyImage.binary_tagging, (seperated,), 'o'))
        self.redraw_operation_stack_tree_view()

    # VISUALISATION
    def visualization_show_menu_bare_command(self):
        imges = self.get_selected_images()
        if len(imges) == 0:
            messagebox.showerror("ERROR", "NO IMAGE WAS PROVIDED")
            return
        ift.plt.clf()
        ift.MyImage.show_images(imges)

    def visualization_show_meta_data_menu_bare_command(self):
        imgs = self.get_selected_images()
        if len(imgs) <= 0:
            messagebox.showerror("ERROR", "NO IMAGE WAS SELECTED")
            return
        if self.meta_data_frame is not None:
            self.meta_data_frame.destroy()

        self.meta_data_frame = MetaDataFrame(self, imgs)
        self.meta_data_frame.run()

    def visualization_show_show_histograms(self):
        imgs = self.get_selected_images()
        if len(imgs) == 0:
            messagebox.showerror("ERROR", "ERROR NO IMAGE WAS SELCTED")
            return

        h = simpledialog.askinteger('HISTOGRAM TYPE',
                                    "CHOOSE THE TYPE OF THE HISTOGRAM\n1- FREQUENCY HISTOGRAM\n2-NORMALIZED "
                                    "HISTOGRAM\n3-CUMULATED HISTOGRAM\n4-CUMULATED NORMALIZED HISTOGRAM")
        if h is None:
            return
        if h not in (1, 2, 3, 4):
            messagebox.showerror('ERROR', "INCORRECT OPTION")
            return

        ift.plt.clf()
        ift.MyImage.show_histograms(imgs, {1: 'h', 2: 'nh', 3: 'ch', 4: "cnh"}[h])

    # Button
    def btn_dlt_command(self):
        selected_images = self.get_selected_images_by_index()
        new_operation_stack = []
        for i, sframe in enumerate(self.operation_stack):
            if i not in selected_images:
                new_operation_stack.append(sframe)

        self.operation_stack = new_operation_stack
        self.redraw_operation_stack_tree_view()

    def btn_cpy_command(self):
        selected_images = self.get_selected_images_by_index()
        new_operation_stack = []
        for i, sframe in enumerate(self.operation_stack):
            if i in selected_images:
                new_operation_stack.append(sframe.copy())
        self.operation_stack.extend(new_operation_stack)
        self.redraw_operation_stack_tree_view()

    def btn_clr_command(self):
        if not messagebox.askyesno("CONFIRMATION", "ARE YOU SURE YOU WANT TO CLEAR THE OPERATIONS STACK ?"): return
        self.operation_stack.clear()
        self.redraw_operation_stack_tree_view()

    def btn_extract_command(self):
        imgs = self.get_selected_images()
        if len(imgs) <= 0:
            messagebox.showerror("ERROR", "ERROR NO IMAGE WAS SELECTED")
            return

        for img in imgs:
            self.operation_stack.append(StackFrame(img, ift.MyImage.copy, (), 'o'))

        self.redraw_operation_stack_tree_view()

    def btn_rename_command(self):
        imgs = self.get_selected_images_by_index()
        if len(imgs) <= 0:
            messagebox.showerror("ERROR", "ERROR NO IMAGE WAS SELECTED")
            return
        if len(imgs) > 1:
            messagebox.showerror("ERROR", "PLEASE SELECT ONE IMAGE ONLY")
            return
        name = simpledialog.askstring('NAME', "ENTER A NAME FOR THE IMAGE or THE GROUP OF IMAGES")
        if name is None:
            return
        self.operation_stack[imgs[0]].name = name
        self.redraw_operation_stack_tree_view()

    # private methods
    def redraw_operation_stack_tree_view(self):
        self.operation_stack_tree_view.delete(*self.operation_stack_tree_view.get_children())
        for i, stck in enumerate(self.operation_stack):
            v = (str(i), stck.function.__name__, str(stck.args), stck.name)
            self.operation_stack_tree_view.insert("", "end", values=v)

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

    def test_image_dimensions(self, img: ift.MyImage) -> bool:
        if img.width > self.MAX_IMG_WIDTH:
            messagebox.showerror("ERROR", f"THE MAXIMUM WIDTH OF AN IMAGE IS {self.MAX_IMG_WIDTH}")
            return False
        if img.height > self.MAX_IMG_HEIGHT:
            messagebox.showerror("ERROR", f"THE MAXIMUM HEIGHT OF AN IMAGE IS {self.MAX_IMG_HEIGHT}")
            return False

        return True

    def test_images_for_unary_operations(self) -> ift.MyImage | None:
        selected_images = self.get_selected_images()
        if len(selected_images) == 0:
            messagebox.showerror("ERROR", "NO IMAGE WAS SELECTED")
            return None

        if len(selected_images) > 1:
            messagebox.showerror("ERROR", "THIS OPERATION TAKE ONLY ONE IMAGE AS INPUT")
            return None
        return selected_images[0]
