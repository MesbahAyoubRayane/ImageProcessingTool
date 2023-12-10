from optparse import Option
from pyclbr import Class
import random
from itertools import groupby
from typing import Generator, Literal, Optional, Self
import numpy as np
import numpy.typing as npty
from PIL import Image
from matplotlib import axis, pyplot as plt



class MyImage: #
    """
    A class which contains all the functions previously learned in the image processing class:
    support two types of images 'RGB' and 'L' (gray scale)
    """
    # typles aliases
    COORDINATES = tuple[int,int]
    RGB_VALUE = tuple[int,int,int]
    RGB_PIXEL = tuple[int,int,int,int,int] # x,y,r,g,b
    GRAY_VALUE = int
    GRAY_PIXEL = tuple[int,int,int] # x,y,g

    MODES = 'RGB', 'L'
    DEFAUL_GRAY_SCALE_COEF = 0.299, 0.587, 0.114 # these are the default values for transforming from rgb to gray scale

    def __init__(self, r: npty.NDArray[np.uint8], g: npty.NDArray[np.uint8], b: npty.NDArray[np.uint8], mode: Literal['RGB','L']):
        """
        r,g,b are numpy matrices they must have the same shape (width*height)
        mode : 'RGB' or 'L'
        """
        mode = mode.strip().upper() # type: ignore
        if mode not in MyImage.MODES:
            raise ValueError(f'Unsupported mode value {mode},mode must be L or RGB')

        if r.ndim != 2 or g.ndim != 2 or b.ndim != 2:
            raise Exception('R,G,B chanels must be in a matrix form')

        if not (r.shape == g.shape and r.shape == b.shape):
            raise Exception('The provided arrays are not coherant')

        if r.size < 4:
            raise Exception("Trying to create an image which has a number of pixels < 4, please create a bigger image")

        for x in r, g, b:
            tmp = x.flatten()
            if (tmp < 0).any():
                raise Exception("The image pixels values must be positive")
            if (tmp > 255).any():
                raise Exception("The imgae pixels values must be < 256")

        self.mode:Literal['RGB','L'] = mode
        self.r:npty.NDArray[np.uint8]  = r.copy().clip(0,255).astype(np.uint8)
        self.g:npty.NDArray[np.uint8]  = g.copy().clip(0,255).astype(np.uint8)
        self.b:npty.NDArray[np.uint8]  = b.copy().clip(0,255).astype(np.uint8)

        # THIS CONSTANTS ARE USED TO IMPROVE COMPUTATION TIME
        self.PIL_IMAGE: Optional[Image.Image] = None  # this will be used in visualization
       
        self.FREQUENCY_HISTOGRAM = None
        self.CUMULATED_FREQUENCY_HISTOGRAM = None
        self.NORMALIZED_FREQUENCY_HISTOGRAM = None
        self.CUMULATED_NORMALIZED_FREQUENCY_HISTOGRAM = None

        # constant value 
        self.MEAN     : Optional[tuple[float,float,float]|float] = None
        self.STD      : Optional[tuple[float,float,float]|float]= None
        self.MEDIAN   : Optional[tuple[float,float,float]|float] = None
        self.OUTLIERS : Optional[tuple[int,int,int]|int] = None

    @property
    def width(self) -> int: return len(self.r[0])

    @property
    def height(self) -> int: return len(self.r)

    @property
    def dimensions(self) -> tuple[int,int]: return self.width, self.height

    def pixels(self) -> Generator[RGB_PIXEL|GRAY_PIXEL,None,None]:
        """
        this function returns all the pixels of the image:
        if mode = rgb  ==> x,y,r,g,b
        if mode = l ==> x,y,v
        """
        if self.mode.upper() == 'RGB':
            for x in range(self.width):
                for y in range(self.height): yield x, y, self.r[y, x], self.g[y, x], self.b[y, x]

        elif self.mode.upper() == 'L':
            for x in range(self.width):
                for y in range(self.height): yield x, y, self.r[y, x]

    def __getitem__(self, indexes: COORDINATES) -> RGB_VALUE|GRAY_VALUE:
        x, y = self.__test_indexes__(indexes[0], indexes[1]) # test if indecies are correct
        if self.mode == 'RGB':
            return int(self.r[y, x]), int(self.g[y, x]), int(self.b[y, x]) 
        elif self.mode == 'L':
            return int(self.r[y,x])
        else:
            raise Exception(f"{self.mode} is unsupported")
        

    def __setitem__(self, indexes: COORDINATES, value: RGB_VALUE|GRAY_VALUE):
        x, y = self.__test_indexes__(indexes[0], indexes[1])
        if self.mode.upper() == "RGB":
            if not isinstance(value, tuple) and not isinstance(value, list):
                raise ValueError(
                    "The provided values are not correct , the image is of type RGB , provide a tuple (R,G,B)")

            r, g, b = int(value[0]), int(value[1]), int(value[2])
            for v in r, g, b:
                if not isinstance(v, int):
                    raise ValueError(f"the provided value is not an integer v={v}")
                if not 0 <= v < 256:
                    raise ValueError('RGB values must be between 0 and 255 inclusive')
            self.r[y, x], self.g[y, x], self.b[y, x] = r, g, b

        elif self.mode.upper() == 'L':
            if isinstance(value, tuple) or isinstance(value, list):
                value = value[0]
            value = int(value)
            if not 0 <= value < 256:
                raise ValueError('RGB values must be between 0 and 255 inclusive')
            self.r[y, x], self.g[y, x], self.b[y, x] = value, value, value
        else:
            raise Exception(f"unsupported mode {self.mode}")

    def __test_indexes__(self, x, y):
        if not 0 <= x < self.width:
            raise Exception(f"x value {x}, is greater than the width of the image {self.width}")
        if not 0 <= y < self.height:
            raise Exception(f"y value {y}, is greater than the height of the image {self.height}")
        return int(x), int(y)

    def copy(self) -> Self:
        # creat a deep copy of the image
        return MyImage(self.r, self.g, self.b, self.mode)

    def cut(self, x: int, y: int, w: int, h: int) -> Self:
        """
        return a sub_image from the original image starting from the point x,y (top-left) to x+w,y+h (bottom-right)
        """
        x, y = self.__test_indexes__(x, y)

        w, h = int(w), int(h)
        if w <= 0 or h <= 0: raise ValueError(f"the width and height must be a positive value , w = {w},h = {h}")
        rimg = MyImage.new(w, h, self.mode)

        for xn in range(w):
            for yn in range(h):
                if x + xn < self.width and y + yn < self.height:
                    rimg[xn, yn] = self[xn + x, yn + y]

        return rimg

    def translation(self, dx: int, dy: int) -> Self:
        dx, dy = int(dx), int(dy)
        r, g, b = self.r.copy(), self.g.copy(), self.b.copy()

        W, H = self.width, self.height
        x_start, y_start = 0, 0
        x_end, y_end = W, H

        if dx > 0:
            r = np.pad(r, pad_width=((0, 0), (dx, 0)), mode='constant', constant_values=0)
            g = np.pad(g, pad_width=((0, 0), (dx, 0)), mode='constant', constant_values=0)
            b = np.pad(b, pad_width=((0, 0), (dx, 0)), mode='constant', constant_values=0)

            x_start = 0
            x_end = W

        elif dx < 0:
            r = np.pad(r, ((0, 0), (0, -dx)), 'constant', constant_values=0)
            g = np.pad(g, ((0, 0), (0, -dx)), 'constant', constant_values=0)
            b = np.pad(b, ((0, 0), (0, -dx)), 'constant', constant_values=0)

            x_start = -dx
            x_end = x_start + W

        if dy > 0:
            r = np.pad(r, ((dy, 0), (0, 0)), 'constant', constant_values=0)
            g = np.pad(g, ((dy, 0), (0, 0)), 'constant', constant_values=0)
            b = np.pad(b, ((dy, 0), (0, 0)), 'constant', constant_values=0)

            y_start = 0
            y_end = H

        elif dy < 0:
            r = np.pad(r, ((0, -dy), (0, 0)), 'constant', constant_values=0)
            g = np.pad(g, ((0, -dy), (0, 0)), 'constant', constant_values=0)
            b = np.pad(b, ((0, -dy), (0, 0)), 'constant', constant_values=0)

            y_start = -dy
            y_end = y_start + H

        r = r[y_start:y_end, x_start:x_end]
        g = g[y_start:y_end, x_start:x_end]
        b = b[y_start:y_end, x_start:x_end]

        return MyImage(r, g, b, self.mode)

    def paste(self, x: int, y: int, w: int, h: int) -> Self:
        """
        This function allow you to paste another image on the new image, the new image must be of a larger size
        for it to contain the old image
        """
        if self.width > w or self.height > h: raise ValueError('The image is bigger than the canvas')
        img = MyImage.new(w, h, self.mode)

        if x < w and y < h:
            yf = min(h - y, self.height)
            xf = min(w - x, self.width)
            img.r[y:y + yf, x:x + xf] = self.r.copy()[:yf, :xf]
            img.g[y:y + yf, x:x + xf] = self.g.copy()[:yf, :xf]
            img.b[y:y + yf, x:x + xf] = self.b.copy()[:yf, :xf]

        return img

    def lay(self, img: Self,mode:Literal["SUM","MAX","MIN","MEAN"]) -> Self:
        """
        take an image as an argument and lay the first image on the second ,resulting on a new image containing both
        Note:
        - If the images have diffrent sizes , the provided image will be scaled to self
        mode:must be one of these options
        SUM  : pixel[x,y] = SUM[self[x,y],img[x,y]]
        MAX  : pixel[x,y] = MAX[self[x,y],img[x,y]]
        MIN  : pixel[x,y] = MIN[self[x,y],img[x,y]]
        MEAN : pixel[x,y] = MEAN[self[x,y],img[x,y]]
        """
        MODES = 'SUM','MAX','MIN','MEAN'
        mode = mode.upper().strip() # type: ignore
        if mode not in MODES: raise ValueError(f"selected overaying mode {mode} is unsupported")
        
        if self.dimensions != img.dimensions:
            # scaling img
            x_fct = self.width / img.width
            y_fct = self.height / img.height
            img = img.rescale(x_fct, y_fct)

        r:npty.NDArray[np.uint8]
        g:npty.NDArray[np.uint8]
        b:npty.NDArray[np.uint8]

        if mode == "SUM":
            r = (self.r.flatten().astype(np.uint32) + img.r.flatten().astype(np.uint32)).clip(0, 255).astype(np.uint8).reshape(self.r.shape)
            g = (self.g.flatten().astype(np.uint32) + img.g.flatten().astype(np.uint32)).clip(0, 255).astype(np.uint8).reshape(self.g.shape)
            b = (self.b.flatten().astype(np.uint32) + img.b.flatten().astype(np.uint32)).clip(0, 255).astype(np.uint8).reshape(self.b.shape)
        
        elif mode == "MAX":
            r = np.concatenate([
                self.r.flatten().reshape((1,-1)),
                img.r.flatten().reshape((1,-1))],
                axis=0).max(axis=0).clip(0,255).astype(np.uint8)
        
            g = np.concatenate([
                self.g.flatten().reshape((1,-1)),
                img.g.flatten().reshape((1,-1))],
                axis=0).max(axis=0).clip(0,255).astype(np.uint8)
        
            b = np.concatenate([
                self.b.flatten().reshape((1,-1)),
                img.b.flatten().reshape((1,-1))],
                axis=0).max(axis=0).clip(0,255).astype(np.uint8)
        
        elif mode =='MIN':
            r = np.concatenate([
                self.r.flatten().reshape((1,-1)),
                img.r.flatten().reshape((1,-1))],
                axis=0).min(axis=0).clip(0,255).astype(np.uint8)
            
            g = np.concatenate([
                self.g.flatten().reshape((1,-1)),
                img.g.flatten().reshape((1,-1))],
                axis=0).min(axis=0).clip(0,255).astype(np.uint8)
            
            b = np.concatenate([
                self.b.flatten().reshape((1,-1)),
                img.b.flatten().reshape((1,-1))],
                axis=0).min(axis=0).clip(0,255).astype(np.uint8)
        
        elif mode == 'MEAN':
            r = (0.5*(self.r.flatten().astype(np.float64) + img.r.flatten().astype(np.float64))).clip(0, 255).astype(np.uint8).reshape(self.r.shape)
            g = (0.5*(self.g.flatten().astype(np.float64) + img.g.flatten().astype(np.float64))).clip(0, 255).astype(np.uint8).reshape(self.g.shape)
            b = (0.5*(self.b.flatten().astype(np.float64) + img.b.flatten().astype(np.float64))).clip(0, 255).astype(np.uint8).reshape(self.b.shape)
        
        else:
            raise ValueError(f"selected overaying mode {mode} is unsupported")

        return MyImage(r.reshape(self.r.shape),g.reshape(self.r.shape),b.reshape(self.r.shape),self.mode)

    def reflect(self, axe: Literal["H","V"]) -> Self:
        axe = axe.lower().strip()  # type: ignore
        if axe not in ('h', 'v'): raise Exception("axe must be v or h")

        if axe == 'v':
            r_img = MyImage(
                np.array([row[::-1] for row in self.r]),
                np.array([row[::-1] for row in self.g]),
                np.array([row[::-1] for row in self.b]), self.mode)
        elif axe == 'h':
            r_img = MyImage(np.array([row for row in self.r[::-1]]), np.array([row for row in self.g[::-1]]),
                            np.array([row for row in self.b[::-1]]), self.mode)
        else:
            raise Exception(f"{self.mode} is not supported")
        return r_img

    def rotate(self, theta: float) -> Self:
        rotated_img = MyImage.new(self.width, self.height, self.mode)
        W, H = rotated_img.width, rotated_img.height
        theta = theta * np.pi / 180
        theta *= -1
        COS, SIN = np.cos(theta), np.sin(theta)
        rotation_matrix_t = np.array([
            [COS, -SIN],
            [SIN, COS]]
        ).transpose()

        U_V = np.array([[i, j] for i in range(W) for j in range(H)]).transpose()
        U_V_MINUS_CENTER = U_V - np.array([[W // 2] * H * W, [H // 2] * H * W])
        X_Y = (rotation_matrix_t @ U_V_MINUS_CENTER) + np.array([[W // 2] * H * W, [H // 2] * H * W])

        U_V = U_V.transpose()
        X_Y = X_Y.transpose()

        for i in range(W * H):
            u, v = U_V[i].tolist()
            x, y = X_Y[i].tolist()
            if (0 <= x < self.width) and (0 <= y < self.height) and (0 <= u < rotated_img.width) and (
                    0 <= v < self.height):
                rotated_img[u, v] = self[x, y]
        return rotated_img

    def rescale(self, x_scaling_factor: float, y_scaling_factor: float) -> Self:
        if x_scaling_factor <= 0 or y_scaling_factor <= 0:
            raise ValueError("The selected factors are incorrect")

        nw = int(self.width * x_scaling_factor)
        nh = int(self.height * y_scaling_factor)

        scaled_img = MyImage.new(nw, nh, self.mode)

        for x in range(nw):
            for y in range(nh):
                u, v = int(x / x_scaling_factor), int(y / y_scaling_factor)
                if (0 <= x < scaled_img.width) and (0 <= y < scaled_img.height) and (0 <= u < self.width) and (
                        0 <= v < self.height):
                    scaled_img[x, y] = self[u, v]

        return scaled_img

    def resolution_under_scaling(self, factor: int) -> Self:
        """
        this function divide the range of each chanel into X bages and affect the mean of each bag to the colors laying inside the range
        exemple:
        factor = 32
        [0:32] [32:64] ... [224:256]
        each pixel laying between 0 and 32 will have the value (0+32)/2
        """
        factor = int(factor)
        if not (0 < factor < 256):
            raise ValueError(f'the factor must bet 0<factor<256 but factor = {factor}')
        if 256 % factor != 0:
            raise ValueError(f"256 must be divisibale by fcator but 256 % {factor} != 0")

        img = MyImage.new(self.width, self.height, self.mode)
        backets = {i // factor: (2*i + factor) // 2 for i in range(0, 256, factor)}  # each backet will map to to a the new color of the backet

        def f(x):
            return backets[x]
        f = np.vectorize(f)

        img.r = f((self.r.flatten() / factor).astype(np.uint32)).astype(np.uint8).reshape(self.r.shape)
        img.g = f((self.g.flatten() / factor).astype(np.uint32)).astype(np.uint8).reshape(self.g.shape)
        img.b = f((self.b.flatten() / factor).astype(np.uint32)).astype(np.uint8).reshape(self.b.shape)

        return img
    
    # histogram based operations
    def histo_translation(self, t: int):
        nr = np.clip(self.r.astype(np.int32) + t, 0, 255).astype(np.uint8)
        ng = np.clip(self.g.astype(np.int32) + t, 0, 255).astype(np.uint8)
        nb = np.clip(self.b.astype(np.int32) + t, 0, 255).astype(np.uint8)
        return MyImage(nr, ng, nb, self.mode)

    def histo_inverse(self):
        r:npty.NDArray[np.uint8] = (255 - self.r.astype(np.int16)).astype(np.uint8)
        g:npty.NDArray[np.uint8] = (255 - self.g.astype(np.int16)).astype(np.uint8)
        b:npty.NDArray[np.uint8] = (255 - self.b.astype(np.int16)).astype(np.uint8)
        return MyImage(r, g, b, self.mode)

    def histo_expansion_dynamique(self) -> Self:
        """
        Note before using this function you need to remove outliers because then can change the results dramaticly
        This function is just a simple normalization function between 0 and 255 , thus outliers have an important effect on the function
        """
        if self.mode == "RGB":
            MIN = self.r.flatten().min()
            MAX = self.r.flatten().max()
            r = np.array((self.r.flatten().astype(np.float64) - MIN) * (255 / (MAX - MIN)), dtype=np.uint8).reshape(
                self.r.shape)
            MIN = self.g.flatten().min()
            MAX = self.g.flatten().max()
            g = np.array((self.g.flatten().astype(np.float64) - MIN) * (255 / (MAX - MIN)), dtype=np.uint8).reshape(
                self.g.shape)
            MIN = self.b.flatten().min()
            MAX = self.b.flatten().max()
            b = np.array((self.b.flatten().astype(np.float64) - MIN) * (255 / (MAX - MIN)), dtype=np.uint8).reshape(
                self.b.shape)
            return MyImage(r, g, b, self.mode)
        elif self.mode == 'L':
            MIN = self.r.flatten().min()
            MAX = self.r.flatten().max()
            gray = np.array((self.r.flatten().astype(np.float64) - MIN) * (255 / (MAX - MIN)), dtype=np.uint8).reshape(
                self.r.shape)
            return MyImage(gray, gray, gray, self.mode)
        else:
            raise Exception(f"{self.mode} is not supported")

    def histo_equalisation(self) -> Self:
        """ use the cumulative histograme to improve contraste"""
        cp_img = self.copy()
        if self.mode == "RGB":
            cdf_r, cdf_g, cdf_b = self.cumulative_normalized_histo()
            for x, y, r, g, b in self.pixels(): # type: ignore
                cp_img[x, y] = (int(cdf_r[int(r)] * 255), int(cdf_g[int(g)] * 255), int(cdf_b[int(b)] * 255))
        
        elif self.mode == 'L':
            cdf: npty.NDArray[np.int32] = self.cumulative_normalized_histo() # type: ignore
            for x, y, v in cp_img.pixels(): # type: ignore
                cp_img[x, y] = int(255 * cdf[int(v)])
        else:
            raise Exception(f"{self.mode} is unsupported")
        return cp_img

    def histo_matching(self, model: Self) -> Self:
        """use an image as a model for another image"""
        if self.mode != model.mode:
            raise ValueError("The selected image model doesn't have the same mode as the modeled image")
        cpyimg = MyImage.new(self.width, self.height, self.mode)
        if self.mode == "L":
            cnh_model: npty.NDArray[np.uint8] = model.cumulative_normalized_histo() # type:ignore
            for x, y, v in self.pixels(): # type:ignore
                cpyimg[x, y] = int(255 * cnh_model[v])

        elif self.mode == "RGB":
            cnh_model_r, cnh_model_g, cnh_model_b = model.cumulative_normalized_histo()
            for x, y, r, g, b in self.pixels(): # type:ignore
                cpyimg[x, y] = (int(255 * cnh_model_r[r]), int(255 * cnh_model_g[g]), int(255 * cnh_model_b[b]))

        else:
            raise Exception(f"{self.mode} is not supported")
        return cpyimg

        # filters

    def gray_scale(self) -> Self:
        coef = MyImage.DEFAUL_GRAY_SCALE_COEF
        Gray = np.array((self.r * coef[0] + self.g * coef[1] + self.b * coef[2]) / sum(coef), dtype=np.uint8)
        return MyImage(Gray, Gray, Gray, 'L')

    def mean_filter(self, size: int) -> Self:
        if isinstance(size, int):
            if size < 2:
                raise ValueError(f'size must be > 2')
            if size % 2 == 0:
                raise ValueError(f"The size must be odd number")
            if size > self.width or size > self.height:
                raise ValueError(f'the provided size is so large')
        else:
            raise ValueError(f"{type(size)} can't be used as a filter")

        copy_img = self.copy()
        kernel = np.full((1, size, size), 1 / (size ** 2))

        r_pad = np.pad(self.r, ((size // 2, size // 2), (size // 2, size // 2)), mode='reflect')
        g_pad = np.pad(self.g, ((size // 2, size // 2), (size // 2, size // 2)), mode='reflect')
        b_pad = np.pad(self.b, ((size // 2, size // 2), (size // 2, size // 2)), mode='reflect')

        r_bag = np.array(
            [r_pad[y - size // 2:y + size // 2 + 1, x - size // 2: x + size // 2 + 1]
             for y in range(size // 2, self.height + size // 2)
             for x in range(size // 2, self.width + size // 2)]
        )
        g_bag = np.array(
            [g_pad[y - size // 2:y + size // 2 + 1, x - size // 2: x + size // 2 + 1]
             for y in range(size // 2, self.height + size // 2)
             for x in range(size // 2, self.width + size // 2)]
        )
        b_bag = np.array(
            [b_pad[y - size // 2:y + size // 2 + 1, x - size // 2: x + size // 2 + 1]
             for y in range(size // 2, self.height + size // 2)
             for x in range(size // 2, self.width + size // 2)]
        )

        copy_img.r = np.clip((r_bag * kernel).sum(axis=(1, 2)), 0, 255).astype(np.uint8).reshape(self.r.shape)
        copy_img.g = np.clip((g_bag * kernel).sum(axis=(1, 2)), 0, 255).astype(np.uint8).reshape(self.r.shape)
        copy_img.b = np.clip((b_bag * kernel).sum(axis=(1, 2)), 0, 255).astype(np.uint8).reshape(self.r.shape)
        return copy_img

    def gaussian_filter(self, size: int, std: float) -> Self:
        if isinstance(size, int):
            if size < 2:
                raise ValueError(f'size must be > 1')
            if size > self.width or size > self.height:
                raise ValueError(f'the provided size is too large')
            if size % 2 == 0:
                raise ValueError(f"size must be odd number")
        else:
            raise ValueError(f"{type(size)} can't be used as a filter")

        # Create a Gaussian kernel using NumPy
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        kernel = np.exp(-((x - size // 2) ** 2 + (y - size // 2) ** 2) / (2 * std ** 2))
        kernel /= (2 * np.pi * std ** 2)
        kernel /= kernel.sum() if kernel.sum() != 0 else 1
        # Normalize the kernel
        # kernel /= kernel.sum() this is not necessary
        # Pad the input image using NumPy
        """
        Padding an image is a common practice in image processing when you want to apply convolution or filtering operations 
        Padding involves adding extra pixels around the edges of the image 
        to ensure that the filter kernel can be applied to all the pixels, even those at the image boundary
        """
        if self.mode == "RGB":
            extended_r = np.pad(self.r, ((size // 2, size // 2), (size // 2, size // 2)), 'reflect')
            extended_g = np.pad(self.g, ((size // 2, size // 2), (size // 2, size // 2)), 'reflect')
            extended_b = np.pad(self.b, ((size // 2, size // 2), (size // 2, size // 2)), 'reflect')

            copy_img = MyImage.new(self.width, self.height, self.mode)

            all_r_patchs = np.array(
                [extended_r[y - size // 2:y + size // 2 + 1, x - size // 2: x + size // 2 + 1]
                 for y in range(size // 2, self.height + size // 2)
                 for x in range(size // 2, self.width + size // 2)
                 ]
            )

            all_g_patchs = np.array(
                [extended_g[y - size // 2:y + size // 2 + 1, x - size // 2:x + size // 2 + 1]
                 for y in range(size // 2, self.height + size // 2)
                 for x in range(size // 2, self.width + size // 2)
                 ]
            )

            all_b_patchs = np.array(
                [extended_b[y - size // 2:y + size // 2 + 1, x - size // 2:x + size // 2 + 1]
                 for y in range(size // 2, self.height + size // 2)
                 for x in range(size // 2, self.width + size // 2)
                 ]
            )

            kernel = kernel.reshape((1, size, size))
            all_r_conv: np.ndarray = np.clip((all_r_patchs * kernel).sum(axis=(1, 2)), 0, 255).reshape(self.r.shape)
            all_g_conv: np.ndarray = np.clip((all_g_patchs * kernel).sum(axis=(1, 2)), 0, 255).reshape(self.r.shape)
            all_b_conv: np.ndarray = np.clip((all_b_patchs * kernel).sum(axis=(1, 2)), 0, 255).reshape(self.r.shape)

            copy_img.r = all_r_conv.astype(np.uint8)
            copy_img.g = all_g_conv.astype(np.uint8)
            copy_img.b = all_b_conv.astype(np.uint8)

            return copy_img

        elif self.mode == "L":
            extended_r = np.pad(self.r, ((size // 2, size // 2), (size // 2, size // 2)), 'reflect')
            copy_img = MyImage.new(self.width, self.height, self.mode)
            all_r_patchs = np.array(
                [extended_r[y - size // 2:y + size // 2 + 1, x - size // 2: x + size // 2 + 1]
                 for y in range(size // 2, self.height + size // 2)
                 for x in range(size // 2, self.width + size // 2)
                 ]
            )
            kernel = kernel.reshape((1, size, size))
            all_r_conv: np.ndarray = np.clip((all_r_patchs * kernel).sum(axis=(1, 2)), 0, 255).reshape(self.r.shape)
            copy_img.r = all_r_conv.astype(np.uint8)
            copy_img.g = copy_img.r
            copy_img.b = copy_img.g
            return copy_img
        else:
            raise ValueError(f"{self.mode} is not supported")

    # TODO if i have time this is a good filter to implement
    def bilateral_filter(self, size: int, std_spatial_gaussian: float, std_brightness_gaussian: float) -> Self:
        size = int(size)
        std_s = float(std_spatial_gaussian)
        std_b = float(std_brightness_gaussian)

        if size < 2:
            raise ValueError(f'size must be > 2')
        if size % 2 == 0:
            raise ValueError(f"The size must be odd number")
        if size > self.width or size > self.height:
            raise ValueError(f'the provided size is so large')

        if std_b <= 0 or std_s <= 0:
            raise ValueError(f"std value must be > 0 {std_s, std_b}")

        X, Y = np.meshgrid(np.arange(size), np.arange(size))
        s_kernel = (np.exp(-0.5 * ((X - size // 2) ** 2 + (Y - size // 2) ** 2) / std_s ** 2) / (
                2 * np.pi * std_s ** 2)).reshape((1, size, size))
        s_kernel /= s_kernel.sum() if s_kernel.sum() != 0 else 1
        cpy_img = MyImage.new(self.width, self.height, self.mode)

        if self.mode == "RGB":
            extended_r = np.pad(self.r, pad_width=size // 2, mode='reflect')
            extended_g = np.pad(self.g, pad_width=size // 2, mode='reflect')
            extended_b = np.pad(self.b, pad_width=size // 2, mode='reflect')

            all_r_patchs = np.array(
                [extended_r[y - size // 2:y + size // 2 + 1, x - size // 2: x + size // 2 + 1]
                 for y in range(size // 2, self.height + size // 2)
                 for x in range(size // 2, self.width + size // 2)
                 ]
            )

            all_g_patchs = np.array(
                [extended_g[y - size // 2:y + size // 2 + 1, x - size // 2:x + size // 2 + 1]
                 for y in range(size // 2, self.height + size // 2)
                 for x in range(size // 2, self.width + size // 2)
                 ]
            )

            all_b_patchs = np.array(
                [extended_b[y - size // 2:y + size // 2 + 1, x - size // 2:x + size // 2 + 1]
                 for y in range(size // 2, self.height + size // 2)
                 for x in range(size // 2, self.width + size // 2)
                 ]
            )

            # gaussian kernels for red
            b_r_kernel = np.array(
                [
                    (extended_r[y - size // 2:y + size // 2 + 1, x - size // 2: x + size // 2 + 1] -
                     np.full((size, size), extended_r[y, x])) ** 2
                    for y in range(size // 2, self.height + size // 2)
                    for x in range(size // 2, self.width + size // 2)
                ]
            )
            b_r_kernel = np.exp(b_r_kernel / np.full((1, size, size), -2 * std_b ** 2)) / (np.sqrt(2 * np.pi) * std_b)

            # gaussian kernels for green
            b_g_kernel = np.array(
                [
                    (extended_g[y - size // 2:y + size // 2 + 1, x - size // 2: x + size // 2 + 1] -
                     np.full((size, size), extended_g[y, x])) ** 2
                    for y in range(size // 2, self.height + size // 2)
                    for x in range(size // 2, self.width + size // 2)
                ]
            )
            b_g_kernel = np.exp(b_g_kernel / np.full((1, size, size), -2 * std_b ** 2)) / (np.sqrt(2 * np.pi) * std_b)

            # gaussian kernels for blue
            b_b_kernel = np.array(
                [
                    (extended_b[y - size // 2:y + size // 2 + 1, x - size // 2: x + size // 2 + 1] -
                     np.full((size, size), extended_b[y, x])) ** 2
                    for y in range(size // 2, self.height + size // 2)
                    for x in range(size // 2, self.width + size // 2)
                ]
            )
            b_b_kernel = np.exp(b_b_kernel / np.full((1, size, size), -2 * std_b ** 2)) / (np.sqrt(2 * np.pi) * std_b)

            # compute the new values of the pixels
            tmp = s_kernel * b_r_kernel
            new_r = (all_r_patchs * tmp).sum(axis=(1, 2)) / (tmp.sum(axis=(1, 2)) if (tmp.sum(axis=(1,2)) != 0).all() else 1)
            tmp = s_kernel * b_g_kernel
            new_g = (all_g_patchs * tmp).sum(axis=(1, 2)) / (tmp.sum(axis=(1, 2)) if (tmp.sum(axis=(1,2)) != 0).all() else 1)
            tmp = s_kernel * b_b_kernel
            new_b = (all_b_patchs * tmp).sum(axis=(1, 2)) / (tmp.sum(axis=(1, 2)) if (tmp.sum(axis=(1,2)) != 0).all() else 1)

            cpy_img.r = np.clip(new_r, 0, 255).astype(np.uint8).reshape(self.r.shape)
            cpy_img.g = np.clip(new_g, 0, 255).astype(np.uint8).reshape(self.r.shape)
            cpy_img.b = np.clip(new_b, 0, 255).astype(np.uint8).reshape(self.r.shape)

            return cpy_img

        elif self.mode == 'L':
            extended_r = np.pad(self.r, pad_width=size // 2, mode='reflect')
            all_r_patchs = np.array(
                [extended_r[y - size // 2:y + size // 2 + 1, x - size // 2: x + size // 2 + 1]
                 for y in range(size // 2, self.height + size // 2)
                 for x in range(size // 2, self.width + size // 2)
                 ]
            )
            b_r_kernel = np.array(
                [
                    (extended_r[y - size // 2:y + size // 2 + 1, x - size // 2: x + size // 2 + 1] -
                     np.full((size, size), extended_r[y, x])) ** 2
                    for y in range(size // 2, self.height + size // 2)
                    for x in range(size // 2, self.width + size // 2)
                ]
            )
            tmp = s_kernel * b_r_kernel
            new_r = (all_r_patchs * tmp).sum(axis=(1, 2)) / (tmp.sum(axis=(1, 2)) if (tmp.sum(axis=(1, 2)) != 0).all() else 1)
            cpy_img.b = cpy_img.g = cpy_img.r = np.clip(new_r, 0, 255).astype(np.uint8).reshape(self.r.shape)
            return cpy_img
        else:
            raise ValueError(f"{self.mode} is not supported")

    def median_filter(self, size: int) -> Self:
        if isinstance(size, int):
            if size < 2:
                raise ValueError(f'size must be > 2')
            if size % 2 == 0:
                raise ValueError(f"The size must be odd number")
            if size > self.width or size > self.height:
                raise ValueError(f'the provided size is so large')
        else:
            raise ValueError(f"{type(size)} can't be used as a filter")

        cpy_img = MyImage.new(self.width, self.height, self.mode)

        if self.mode == "RGB":
            pad_r = np.pad(self.r, size // 2, "reflect")
            pad_g = np.pad(self.g, size // 2, "reflect")
            pad_b = np.pad(self.b, size // 2, "reflect")

            r_bag = np.array(
                [pad_r[y - size // 2:y + size // 2 + 1, x - size // 2: x + size // 2 + 1]
                 for y in range(size // 2, self.height + size // 2)
                 for x in range(size // 2, self.width + size // 2)
                 ]
            )
            g_bag = np.array(
                [pad_g[y - size // 2:y + size // 2 + 1, x - size // 2: x + size // 2 + 1]
                 for y in range(size // 2, self.height + size // 2)
                 for x in range(size // 2, self.width + size // 2)
                 ]
            )
            b_bag = np.array([
                pad_b[y - size // 2:y + size // 2 + 1, x - size // 2: x + size // 2 + 1]
                for y in range(size // 2, self.height + size // 2)
                for x in range(size // 2, self.width + size // 2)
            ]
            )

            cpy_img.r = np.median(r_bag, axis=(1, 2)).reshape(self.r.shape).astype(np.uint8)
            cpy_img.g = np.median(g_bag, axis=(1, 2)).reshape(self.r.shape).astype(np.uint8)
            cpy_img.b = np.median(b_bag, axis=(1, 2)).reshape(self.r.shape).astype(np.uint8)

            return cpy_img

        elif self.mode == 'L':
            pad_r = np.pad(self.r, size // 2, "reflect")
            r_bag = np.array(
                [pad_r[y - size // 2:y + size // 2 + 1, x - size // 2: x + size // 2 + 1]
                 for y in range(size // 2, self.height + size // 2)
                 for x in range(size // 2, self.width + size // 2)
                 ]
            )
            cpy_img.r = np.median(r_bag, axis=(1, 2)).reshape(self.r.shape).astype(np.uint8)
            cpy_img.g = cpy_img.b = cpy_img.r
            return cpy_img
        else:
            raise ValueError(f"{self.mode} is not supported")

    def min_filter(self, size: int) -> Self:
        if isinstance(size, int):
            if size < 2:
                raise ValueError(f'size must be > 2')
            if size % 2 == 0:
                raise ValueError(f"The size must be odd number")
            if size > self.width or size > self.height:
                raise ValueError(f'the provided size is so large')
        else:
            raise ValueError(f"{type(size)} can't be used as a filter")

        cpy_img = MyImage.new(self.width, self.height, self.mode)

        if self.mode == "RGB":
            pad_r = np.pad(self.r, size // 2, "reflect")
            pad_g = np.pad(self.g, size // 2, "reflect")
            pad_b = np.pad(self.b, size // 2, "reflect")

            r_bag = np.array(
                [pad_r[y - size // 2:y + size // 2 + 1, x - size // 2: x + size // 2 + 1]
                 for y in range(size // 2, self.height + size // 2)
                 for x in range(size // 2, self.width + size // 2)
                 ]
            )
            g_bag = np.array(
                [pad_g[y - size // 2:y + size // 2 + 1, x - size // 2: x + size // 2 + 1]
                 for y in range(size // 2, self.height + size // 2)
                 for x in range(size // 2, self.width + size // 2)
                 ]
            )
            b_bag = np.array([
                pad_b[y - size // 2:y + size // 2 + 1, x - size // 2: x + size // 2 + 1]
                for y in range(size // 2, self.height + size // 2)
                for x in range(size // 2, self.width + size // 2)
            ]
            )

            cpy_img.r = np.min(r_bag, axis=(1, 2)).reshape(self.r.shape).astype(np.uint8)
            cpy_img.g = np.min(g_bag, axis=(1, 2)).reshape(self.r.shape).astype(np.uint8)
            cpy_img.b = np.min(b_bag, axis=(1, 2)).reshape(self.r.shape).astype(np.uint8)

            return cpy_img

        elif self.mode == 'L':
            pad_r = np.pad(self.r, size // 2, "reflect")
            r_bag = np.array(
                [pad_r[y - size // 2:y + size // 2 + 1, x - size // 2: x + size // 2 + 1]
                 for y in range(size // 2, self.height + size // 2)
                 for x in range(size // 2, self.width + size // 2)
                 ]
            )
            cpy_img.r = np.min(r_bag, axis=(1, 2)).reshape(self.r.shape).astype(np.uint8)
            cpy_img.g = cpy_img.b = cpy_img.r
            return cpy_img
        else:
            raise ValueError(f"{self.mode} is not supported")

    def max_filter(self, size: int) -> Self:
        if isinstance(size, int):
            if size < 2:
                raise ValueError(f'size must be > 2')
            if size % 2 == 0:
                raise ValueError(f"The size must be odd number")
            if size > self.width or size > self.height:
                raise ValueError(f'the provided size is so large')
        else:
            raise ValueError(f"{type(size)} can't be used as a filter")

        cpy_img = MyImage.new(self.width, self.height, self.mode)

        if self.mode == "RGB":
            pad_r = np.pad(self.r, size // 2, "reflect")
            pad_g = np.pad(self.g, size // 2, "reflect")
            pad_b = np.pad(self.b, size // 2, "reflect")

            r_bag = np.array(
                [pad_r[y - size // 2:y + size // 2 + 1, x - size // 2: x + size // 2 + 1]
                 for y in range(size // 2, self.height + size // 2)
                 for x in range(size // 2, self.width + size // 2)
                 ]
            )
            g_bag = np.array(
                [pad_g[y - size // 2:y + size // 2 + 1, x - size // 2: x + size // 2 + 1]
                 for y in range(size // 2, self.height + size // 2)
                 for x in range(size // 2, self.width + size // 2)
                 ]
            )
            b_bag = np.array([
                pad_b[y - size // 2:y + size // 2 + 1, x - size // 2: x + size // 2 + 1]
                for y in range(size // 2, self.height + size // 2)
                for x in range(size // 2, self.width + size // 2)
            ]
            )
            cpy_img.r = np.max(r_bag, axis=(1, 2)).reshape(self.r.shape).astype(np.uint8)
            cpy_img.g = np.max(g_bag, axis=(1, 2)).reshape(self.r.shape).astype(np.uint8)
            cpy_img.b = np.max(b_bag, axis=(1, 2)).reshape(self.r.shape).astype(np.uint8)

        elif self.mode == 'L':
            pad_r = np.pad(self.r, size // 2, "reflect")
            r_bag = np.array(
                [pad_r[y - size // 2:y + size // 2 + 1, x - size // 2: x + size // 2 + 1]
                 for y in range(size // 2, self.height + size // 2)
                 for x in range(size // 2, self.width + size // 2)
                 ]
            )
            cpy_img.r = np.max(r_bag, axis=(1, 2)).reshape(self.r.shape).astype(np.uint8)
            cpy_img.g = cpy_img.b = cpy_img.r
        else:
            raise ValueError(f"{self.mode} is not supported")
        return cpy_img

    def laplacian_sharpning_filter(self, distance: str, size: int) -> Self:
        """
        distance must be on of these variants : MANHATTAN,MAX
        size : is an odd positive number
        """
        size = int(size)
        if size < 2: raise ValueError('size must be > 2')
        if size % 2 == 0: raise ValueError('size must be odd number')

        distance = distance.lower().strip()
        if distance == 'manhattan':
            kernel = np.zeros((size, size))
            kernel[size // 2, :] = 1
            kernel[:, size // 2] = 1
            kernel[size // 2, size // 2] = -(kernel.sum() - 1)
        elif distance == "max":
            kernel = np.ones((size, size))
            kernel[size // 2, size // 2] = -(kernel.sum() - 1)
        else:
            raise ValueError("distance must be 4 or 8")
        kernel = kernel.reshape((1, size, size))

        r_pad = np.pad(self.r, size // 2, 'reflect')
        g_pad = np.pad(self.g, size // 2, 'reflect')
        b_pad = np.pad(self.b, size // 2, 'reflect')
        copy_img = MyImage.new(self.width, self.height, self.mode)
        r_bag = np.array(
            [r_pad[y - size // 2:y + size // 2 + 1, x - size // 2: x + size // 2 + 1]
             for y in range(size // 2, self.height + size // 2)
             for x in range(size // 2, self.width + size // 2)]
        )
        g_bag = np.array(
            [g_pad[y - size // 2:y + size // 2 + 1, x - size // 2: x + size // 2 + 1]
             for y in range(size // 2, self.height + size // 2)
             for x in range(size // 2, self.width + size // 2)]
        )
        b_bag = np.array(
            [b_pad[y - size // 2:y + size // 2 + 1, x - size // 2: x + size // 2 + 1]
             for y in range(size // 2, self.height + size // 2)
             for x in range(size // 2, self.width + size // 2)]
        )

        copy_img.r = np.clip((r_bag * kernel).sum(axis=(1, 2)), 0, 255).astype(np.uint8).reshape(self.r.shape)
        copy_img.g = np.clip((g_bag * kernel).sum(axis=(1, 2)), 0, 255).astype(np.uint8).reshape(self.r.shape)
        copy_img.b = np.clip((b_bag * kernel).sum(axis=(1, 2)), 0, 255).astype(np.uint8).reshape(self.r.shape)

        return copy_img

    def edge_detection_robert(self, threshold: int) -> Self:
        kernel_diag = np.array([[-1, 0], [0, 1]]).reshape((1, 2, 2))
        kernel_rev_diag = np.array([[0, -1], [1, 0]]).reshape((1, 2, 2))

        extended_r = np.pad(self.r, pad_width=1, mode='reflect')
        extended_g = np.pad(self.g, pad_width=1, mode='reflect')
        extended_b = np.pad(self.b, pad_width=1, mode='reflect')

        W, H = self.width, self.height

        bage_r = np.array([
            extended_r[y:y + 2, x:x + 2]
            for y in range(1, H + 1)
            for x in range(1, W + 1)
        ])
        bage_g = np.array([
            extended_g[y:y + 2, x:x + 2]
            for y in range(1, H + 1)
            for x in range(1, W + 1)
        ])
        bage_b = np.array([
            extended_b[y:y + 2, x:x + 2]
            for y in range(1, H + 1)
            for x in range(1, W + 1)
        ])
        G_diag_r = (bage_r * kernel_diag).sum(axis=(1, 2))
        G_diag_g = (bage_g * kernel_diag).sum(axis=(1, 2))
        G_diag_b = (bage_b * kernel_diag).sum(axis=(1, 2))

        G_rev_diag_r = (bage_r * kernel_rev_diag).sum(axis=(1, 2))
        G_rev_diag_g = (bage_g * kernel_rev_diag).sum(axis=(1, 2))
        G_rev_diag_b = (bage_b * kernel_rev_diag).sum(axis=(1, 2))

        G_r: np.ndarray = np.sqrt(G_diag_r ** 2 + G_rev_diag_r ** 2)
        G_g: np.ndarray = np.sqrt(G_diag_g ** 2 + G_rev_diag_g ** 2)
        G_b: np.ndarray = np.sqrt(G_diag_b ** 2 + G_rev_diag_b ** 2)

        G_r: np.ndarray = (G_r - G_r.min()) / (G_r.max() - G_r.min()) * 255
        G_g: np.ndarray = (G_g - G_g.min()) / (G_g.max() - G_g.min()) * 255
        G_b: np.ndarray = (G_b - G_b.min()) / (G_b.max() - G_b.min()) * 255

        f = np.vectorize(lambda x: 255 if x > threshold else 0)
        G_r = f(G_r)
        G_g = f(G_g)
        G_b = f(G_b)

        G_r = G_r.astype(np.uint8).reshape(self.r.shape)
        G_g = G_g.astype(np.uint8).reshape(self.r.shape)
        G_b = G_b.astype(np.uint8).reshape(self.r.shape)

        return MyImage(G_r, G_g, G_b, self.mode)

    def edge_detection_sobel(self, threshold: int) -> Self:
        """
        threshold is an integer value between 0 and 255 , the lower it is the more daitaills will be detected as an edge
        """
        kernel_h = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]).reshape((1, 3, 3))
        kernel_v = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ]).reshape((1, 3, 3))

        W, H = self.width, self.height

        extended_r = np.pad(self.r, 1, 'reflect')
        extended_g = np.pad(self.g, 1, 'reflect')
        extended_b = np.pad(self.b, 1, 'reflect')
        # creating the bags
        bage_r = np.array([
            extended_r[y - 1:y + 2, x - 1:x + 2]
            for y in range(1, H + 1)
            for x in range(1, W + 1)
        ])
        bage_g = np.array([
            extended_g[y - 1:y + 2, x - 1:x + 2]
            for y in range(1, H + 1)
            for x in range(1, W + 1)
        ])
        bage_b = np.array([
            extended_b[y - 1:y + 2, x - 1:x + 2]
            for y in range(1, H + 1)
            for x in range(1, W + 1)
        ])
        G_r_h = (bage_r * kernel_h).sum(axis=(1, 2))
        G_r_v = (bage_r * kernel_v).sum(axis=(1, 2))
        G_g_h = (bage_g * kernel_h).sum(axis=(1, 2))
        G_g_v = (bage_g * kernel_v).sum(axis=(1, 2))
        G_b_h = (bage_b * kernel_h).sum(axis=(1, 2))
        G_b_v = (bage_b * kernel_v).sum(axis=(1, 2))

        G_r = np.sqrt(G_r_h ** 2 + G_r_v ** 2)
        G_g = np.sqrt(G_g_h ** 2 + G_g_v ** 2)
        G_b = np.sqrt(G_b_h ** 2 + G_b_v ** 2)

        # normilizing the gradiants from 0..255
        G_r = (G_r - G_r.min()) / (G_r.max() - G_r.min()) * 255
        G_g = (G_g - G_g.min()) / (G_g.max() - G_g.min()) * 255
        G_b = (G_b - G_b.min()) / (G_b.max() - G_b.min()) * 255

        f = np.vectorize(lambda x: 255 if x > threshold else 0)
        G_r: np.ndarray = f(G_r)
        G_g: np.ndarray = f(G_g)
        G_b: np.ndarray = f(G_b)

        G_r = G_r.astype(np.uint8).reshape(self.r.shape)
        G_g = G_g.astype(np.uint8).reshape(self.r.shape)
        G_b = G_b.astype(np.uint8).reshape(self.r.shape)

        return MyImage(G_r, G_g, G_b, self.mode)

    def edges_detection_prewitt(self, threshold: int) -> Self:
        """
        threshold is an integer value between 0 and 255 , the lower it is the more daitaills will be detected as an edge
        """
        kernel_h = np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ]).reshape((1, 3, 3))
        kernel_v = np.array([
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ]).reshape((1, 3, 3))

        W, H = self.width, self.height

        extended_r = np.pad(self.r, 1, 'reflect')
        extended_g = np.pad(self.g, 1, 'reflect')
        extended_b = np.pad(self.b, 1, 'reflect')
        # creating the bags
        bage_r = np.array([
            extended_r[y - 1:y + 2, x - 1:x + 2]
            for y in range(1, H + 1)
            for x in range(1, W + 1)
        ])
        bage_g = np.array([
            extended_g[y - 1:y + 2, x - 1:x + 2]
            for y in range(1, H + 1)
            for x in range(1, W + 1)
        ])
        bage_b = np.array([
            extended_b[y - 1:y + 2, x - 1:x + 2]
            for y in range(1, H + 1)
            for x in range(1, W + 1)
        ])
        G_r_h = (bage_r * kernel_h).sum(axis=(1, 2))
        G_r_v = (bage_r * kernel_v).sum(axis=(1, 2))
        G_g_h = (bage_g * kernel_h).sum(axis=(1, 2))
        G_g_v = (bage_g * kernel_v).sum(axis=(1, 2))
        G_b_h = (bage_b * kernel_h).sum(axis=(1, 2))
        G_b_v = (bage_b * kernel_v).sum(axis=(1, 2))

        G_r = np.sqrt(G_r_h ** 2 + G_r_v ** 2)
        G_g = np.sqrt(G_g_h ** 2 + G_g_v ** 2)
        G_b = np.sqrt(G_b_h ** 2 + G_b_v ** 2)

        # normilizing the gradiants from 0..255
        G_r = (G_r - G_r.min()) / (G_r.max() - G_r.min()) * 255
        G_g = (G_g - G_g.min()) / (G_g.max() - G_g.min()) * 255
        G_b = (G_b - G_b.min()) / (G_b.max() - G_b.min()) * 255

        f = np.vectorize(lambda x: 255 if x > threshold else 0)
        G_r: np.ndarray = f(G_r)
        G_g: np.ndarray = f(G_g)
        G_b: np.ndarray = f(G_b)

        G_r = G_r.astype(np.uint8).reshape(self.r.shape)
        G_g = G_g.astype(np.uint8).reshape(self.r.shape)
        G_b = G_b.astype(np.uint8).reshape(self.r.shape)

        return MyImage(G_r, G_g, G_b, self.mode)

    def kmean(self, k: int) -> list[Self]:
        COLOR_CHANEL = np.ndarray

        def _kmean(r_chanel: COLOR_CHANEL, g_chanel: COLOR_CHANEL, b_chanel: COLOR_CHANEL, k: int) -> list[list]:
            SIZE: int = len(r_chanel)
            clusters: dict[tuple[int, int, int], set[int]] = {
                (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)): set()
                for _ in range(k)
            }
            cnt = True
            while cnt:
                cnt = False
                centers: COLOR_CHANEL = np.array(list(clusters.keys()))
                distances = np.sqrt(np.power(centers[:, 0].reshape((-1, 1)) - r_chanel, 2) + \
                                    np.power(centers[:, 1].reshape((-1, 1)) - g_chanel, 2) + \
                                    np.power(centers[:, 2].reshape((-1, 1)) - b_chanel, 2))
                neighrest_cluster = distances.argsort(axis=0)[0]

                index_key_mapper = {
                    i: x for (i, x) in enumerate(clusters.keys())
                }
                new_clusters = {k: set() for k in clusters.keys()}
                for i, nc in enumerate(neighrest_cluster):
                    new_clusters[index_key_mapper[nc]].add(i)
                for (new, old) in zip(new_clusters.values(), clusters.values()):
                    if old != new:
                        cnt = True
                        break

                # recomputing the centers of the clusters
                clusters.clear()
                for center, cluster in new_clusters.items():
                    cluster = np.arange(SIZE)[list(cluster)]
                    if len(cluster) == 0:
                        clusters[center] = set()
                        continue
                    new_center = r_chanel[cluster].mean(), g_chanel[cluster].mean(), b_chanel[cluster].mean()
                    clusters[new_center] = set(cluster.tolist())
            return [list(x) for x in clusters.values()]

        r_chanel = self.r.flatten()
        g_chanel = self.g.flatten()
        b_chanel = self.b.flatten()
        clusters = _kmean(r_chanel, g_chanel, b_chanel, k)
        imgs = []
        for cluster in clusters:
            img = MyImage.new(self.width, self.height, self.mode)
            for i in cluster:
                x, y = int(i % self.width), i // self.width
                img[x, y] = self[x, y]
            imgs.append(img)
        return imgs

    def segmentation_by_threshold(self, threshold: int|tuple[int,int,int]) -> list[Self]:
        if self.mode == 'L':
            if not (0 <= threshold < 256): # type:ignore
                raise ValueError("Threshold must be between 0 and 255")
        elif self.mode == 'RGB':
            threshold_r,threshold_g,threshold_b = threshold # type:ignore
            if not (0 <= threshold_r < 256):
                raise ValueError("R Threshold must be between 0 and 255")
            if not (0 <= threshold_g < 256):
                raise ValueError("G Threshold must be between 0 and 255")
            if not (0 <= threshold_b < 256):
                raise ValueError("B Threshold must be between 0 and 255")

        img0 = MyImage.new(self.width, self.height, self.mode)
        img1 = MyImage.new(self.width, self.height, self.mode)

        if self.mode == 'RGB':
            img0.r[self.r > threshold_r] = self.r[self.r > threshold_r].copy() # type:ignore
            img0.g[self.g > threshold_g] = self.g[self.g > threshold_g].copy() # type:ignore
            img0.b[self.b > threshold_b] = self.b[self.b > threshold_b].copy() # type:ignore

            img1.r[self.r <= threshold_r] = self.r[self.r <= threshold_r].copy() # type:ignore
            img1.g[self.g <= threshold_g] = self.g[self.g <= threshold_g].copy() # type:ignore
            img1.b[self.b <= threshold_b] = self.b[self.b <= threshold_b].copy() # type:ignore

        elif self.mode == 'L':
            img0.r[self.r > threshold] = self.r[self.r > threshold].copy()
            img0.b = img0.g = img0.r

            img1.r[self.r <= threshold] = self.r[self.r <= threshold].copy()
            img1.g = img1.b = img1.r

        else: raise Exception(f"{self.mode} is unsupported")

        return [img0, img1]

    def binary_tagging(self, seperated: bool) -> Self | list[Self]:
        def get_neighbores(x: int, y: int): return [(x-1,y-1),(x-1,y),(x-1,y+1),(x,y-1),(x,y+1),(x+1,y-1),(x+1,y),(x+1,y+1)]
        
        tag_matrix: np.ndarray = np.zeros(self.r.shape)
        colored: np.ndarray = (self.r > 0) | (self.g > 0) | (self.b > 0)
        tag = 1
        
        for x in range(self.width):
            for y in range(self.height):
                if tag_matrix[y, x] != 0 or colored[y, x] == False: continue
                tag_matrix[y, x] = tag
                neighbores = get_neighbores(x, y)
                while len(neighbores) > 0:
                    xn, yn = neighbores.pop()
                    if not (0 <= xn < self.width and 0 <= yn < self.height): continue

                    # if the pixel is already taged or the pixel is not colored
                    # note this condition (tag_matrix[yn, xn] != 0) can be changed just to tag_matrix[yn, xn] != tag
                    if tag_matrix[yn, xn] != 0 or (colored[yn, xn] == False): continue
                    tag_matrix[yn, xn] = tag
                    neighbores.extend(set(get_neighbores(xn, yn)))
                tag += 1

        if seperated:
            imgs = {t: MyImage.new(self.width, self.height, self.mode) for t in range(1, tag)}
            for x in range(self.width):
                for y in range(self.height):
                    if tag_matrix[y, x] == 0: continue
                    imgs[int(tag_matrix[y, x])].r[y, x] = self.r[y, x]
                    imgs[int(tag_matrix[y, x])].g[y, x] = self.g[y, x]
                    imgs[int(tag_matrix[y, x])].b[y, x] = self.g[y, x]

            return list(imgs.values())
        else:
            colors = set()
            while len(colors) < tag:
                colors.add((np.random.randint(0, 256, dtype=np.uint8),
                            np.random.randint(0, 256, dtype=np.uint8),
                            np.random.randint(0, 256, dtype=np.uint8)))
            colors = list(colors)
            img = MyImage.new(self.width, self.height, self.mode)
            
            for x in range(self.width):
                for y in range(self.height):
                    if tag_matrix[y, x] == 0: continue
                    r, g, b = colors[int(tag_matrix[y, x])]
                    img.r[y, x] = r
                    img.g[y, x] = g
                    img.b[y, x] = b
            return img

    # histogrames
    def histograme(self) -> npty.NDArray | tuple[npty.NDArray, npty.NDArray, npty.NDArray]:
        if self.FREQUENCY_HISTOGRAM is None:
            if self.mode == 'L':
                histo = np.full(256, 0)
                pxl = self.r.flatten()
                pxl.sort()
                for p, r in groupby(pxl):
                    histo[int(p)] = len(list(r))
                self.FREQUENCY_HISTOGRAM = histo

            elif self.mode == 'RGB':
                result = []
                for x in self.r, self.g, self.b:
                    histo = np.full(256, 0)
                    pxl = x.flatten()
                    pxl.sort()
                    for p in groupby(pxl):
                        p, r = int(p[0]), len(list(p[1]))
                        histo[p] = r
                    result.append(histo)
                self.FREQUENCY_HISTOGRAM = np.array(result[0]), np.array(result[1]), np.array(result[2])

            else:
                raise ValueError(f"{self.mode} is not supported")

        return self.FREQUENCY_HISTOGRAM

    def cumulated_histograme(self) -> npty.NDArray| tuple[npty.NDArray, npty.NDArray, npty.NDArray]:
        if self.CUMULATED_FREQUENCY_HISTOGRAM is None:
            if self.mode == "RGB":
                hr, hg, hb = self.histograme()
                chr, chg, chb = np.full((256,), dtype=np.int32, fill_value=0), \
                    np.full((256,), dtype=np.int32, fill_value=0), \
                    np.full((256,), dtype=np.int32, fill_value=0)
                sum_r = sum_g = sum_b = 0
                for i in range(256):
                    sum_r += hr[i]
                    sum_g += hg[i]
                    sum_b += hb[i]
                    chr[i] = sum_r
                    chg[i] = sum_g
                    chb[i] = sum_b

                self.CUMULATED_FREQUENCY_HISTOGRAM = chr, chg, chb  # type: ignore

            elif self.mode == 'L':
                hgray = self.histograme()
                chgray = np.full((256,), dtype=np.int32, fill_value=0)
                sum_gray = 0
                for i in range(256):
                    sum_gray += hgray[i]
                    chgray[i] = sum_gray
                self.CUMULATED_FREQUENCY_HISTOGRAM = chgray

            else:
                raise ValueError(f"{self.mode} is not supported")

        return self.CUMULATED_FREQUENCY_HISTOGRAM

    def normalized_histograme(self) -> npty.NDArray | tuple[npty.NDArray, npty.NDArray, npty.NDArray]:
        if self.NORMALIZED_FREQUENCY_HISTOGRAM is None:
            if self.mode == "RGB":
                hr, hg, hb = self.histograme()
                nhr, nhg, nhb = np.array(hr, dtype=np.float64), np.array(hg, dtype=np.float64), np.array(hb,
                                                                                                         dtype=np.float64)
                nhr /= (self.width * self.height)
                nhg /= (self.width * self.height)
                nhb /= (self.width * self.height)
                self.NORMALIZED_FREQUENCY_HISTOGRAM = nhr, nhg, nhb

            elif self.mode == 'L':
                hgray = self.histograme()
                nhgray = np.array(hgray, dtype=np.float64) / (self.width * self.height)
                self.NORMALIZED_FREQUENCY_HISTOGRAM = nhgray

            else:
                raise Exception(f"{self.mode} is not supported")

        return self.NORMALIZED_FREQUENCY_HISTOGRAM

    def cumulative_normalized_histo(self) -> npty.NDArray | tuple[npty.NDArray, npty.NDArray, npty.NDArray]:
        if self.CUMULATED_NORMALIZED_FREQUENCY_HISTOGRAM is None:
            if self.mode == 'RGB':
                nhr, nhg, nhb = self.normalized_histograme()
                cnhr, cnhg, cnhb = np.full(256, 0.), np.full(256, 0.), np.full(256, 0.)
                cnhr[0] = nhr[0]
                cnhg[0] = nhg[0]
                cnhb[0] = nhb[0]
                for i in range(1, 256):
                    cnhr[i] = nhr[i] + cnhr[i - 1]
                    cnhg[i] = nhg[i] + cnhg[i - 1]
                    cnhb[i] = nhb[i] + cnhb[i - 1]
                self.CUMULATED_NORMALIZED_FREQUENCY_HISTOGRAM = cnhr, cnhg, cnhb

            elif self.mode == 'L':
                nhgray = self.normalized_histograme()
                cnhgray = np.full(256, 0.)
                cnhgray[0] = nhgray[0]
                for i in range(1, 256):
                    cnhgray[i] = nhgray[i] + cnhgray[i - 1]
                self.CUMULATED_NORMALIZED_FREQUENCY_HISTOGRAM = cnhgray

            else:
                raise Exception(f"{self.mode} is not supported")

        return self.CUMULATED_NORMALIZED_FREQUENCY_HISTOGRAM

        # statistical constants

    def mean(self) -> tuple[float, float, float] | float:
        if self.MEAN is None:
            if self.mode == "RGB":
                self.MEAN = float(self.r.flatten().mean()), float(self.g.flatten().mean()), float(self.b.flatten().mean())
            elif self.mode == "L":
                self.MEAN = float(self.r.flatten().mean())
            else:
                raise Exception(f"{self.mode} is unsupported mode")
        return self.MEAN

    def std(self) -> tuple[float, float, float] | float:
        if self.STD is None:
            if self.mode == "RGB":
                self.STD = self.r.flatten().std(), self.g.flatten().std(), self.b.flatten().std()
            elif self.mode == "L":
                self.STD = self.r.flatten().std()
            else:
                raise Exception(f"{self.mode} is unsupported mode")
        return self.STD

    def median(self) -> tuple[float, float, float] | float:
        if self.MEDIAN is None:
            if self.mode == "RGB":
                self.MEDIAN = np.median(self.r.flatten()), np.median(self.g.flatten()), np.median(self.b.flatten())
            elif self.mode == "L":
                self.MEDIAN = np.median(self.r.flatten())
            else:
                raise Exception(f"{self.mode} is unsupported mode")
        return self.MEDIAN

    def outliers(self) -> tuple[int, int, int] | int:
        if self.OUTLIERS is None:
            if self.mode == "RGB":
                r_mean = self.r.flatten().mean()
                r_std = self.r.std()
                r_out = ((self.r.flatten() > r_mean + 1.5 * r_std) | (self.r.flatten() < r_mean - 1.5 * r_std)).sum()

                g_mean = self.g.flatten().mean()
                g_std = self.g.flatten().std()
                g_out = ((self.g.flatten() > g_mean + 1.5 * g_std) | (self.g.flatten() < g_mean - 1.5 * g_std)).sum()

                b_mean = self.b.flatten().mean()
                b_std = self.b.flatten().std()
                b_out = ((self.b.flatten() > b_mean + 1.5 * b_std) | (self.b.flatten() < b_mean - 1.5 * b_std)).sum()

                self.OUTLIERS = int(r_out), int(g_out), int(b_out)

            elif self.mode == 'L':
                gray_mean = self.r.flatten().mean()
                gray_std = self.r.std()
                gray_out = ((self.r.flatten() > gray_mean + 1.5 * gray_std) | (
                        self.r.flatten() < gray_mean - 1.5 * gray_std)).sum()
                self.OUTLIERS = int(gray_out)
            else:
                raise Exception(f"{self.mode} is not supported")
        return self.OUTLIERS

    def index(self) -> npty.NDArray:
        if self.mode == 'RGB':
            A, B, C = self.normalized_histograme()
            a, b, c = self.cut(0, 0, self.width // 2, self.height // 2).normalized_histograme()
            d, e, f = self.cut(self.width // 2, 0, self.width // 2, self.height // 2).normalized_histograme()
            g, h, i = self.cut(0, self.height // 2, self.width // 2, self.height // 2).normalized_histograme()
            j, k, l = self.cut(self.width // 2, self.height // 2, self.width // 2,
                               self.height // 2).normalized_histograme()

        elif self.mode == 'L':
            A = B = C = self.normalized_histograme()
            a = b = c = self.cut(0, 0, self.width // 2, self.height // 2).normalized_histograme()
            d = e = f = self.cut(self.width // 2, 0, self.width // 2, self.height // 2).normalized_histograme()
            g = h = i = self.cut(0, self.height // 2, self.width // 2, self.height // 2).normalized_histograme()
            j = k = l = self.cut(self.width // 2, self.height // 2, self.width // 2,
                                 self.height // 2).normalized_histograme()

        else:
            raise Exception(f"{self.mode} is not supported")

        return np.concatenate([A, B, C, a, b, c, d, e, f, g, h, i, j, k, l])

    @staticmethod
    def new(w: int, h: int, mode: Literal['RGB','L']):
        """
        create a new image having width w and hight h , and initilise the rgb matrices to zero 
        """
        mode = mode.upper() # type: ignore
        if mode not in MyImage.MODES: raise ValueError(f'the selected mode <{mode}> is not provided')
        v = np.full((h, w), 0, dtype=np.uint8)
        return MyImage(v, v, v, mode)

    @staticmethod
    def new_from_pixels(pixels: list[RGB_PIXEL|GRAY_PIXEL], mode: Literal["RGB","L"], width: int, height: int):
        if len(pixels) == 0:
            raise ValueError("no pixels were given")
        img = MyImage.new(width, height, mode)
        for item in pixels:
            x, y, *v = item
            img[x, y] = v # type: ignore
        return img

    @staticmethod
    def open_image_as_rgb_matrices(path: str):
        """
            Read an image from a file described by the provided path and returns a tuple\n
            which contains 3 matrices representing the values of the image respectevly  R,G,B 
        """
        img = Image.open(path).convert("RGB")
        R, G, B = [], [], []

        for r, g, b in img.getdata():
            R.append(r) 
            G.append(g)
            B.append(b)

        R = np.array(R).reshape((img.height, img.width))
        G = np.array(G).reshape((img.height, img.width))
        B = np.array(B).reshape((img.height, img.width))

        return MyImage(R, G, B, 'RGB')

    def save_image(self, path: str):
        img_to_save = Image.new(self.mode, self.dimensions)

        if self.mode == 'L':
            for x in range(self.width):
                for y in range(self.height):
                    v = self[x,y]
                    img_to_save.putpixel((x, y), v)
        elif self.mode == 'RGB':
            for x in range(self.width):
                for y in range(self.height):
                    img_to_save.putpixel((x, y), self[x, y])
        else:
            raise Exception(f"{self.mode} is not supported")

        img_to_save.save(path)

    # VISUALISATION FUNCTIONS
    def show_image(self):
        plt.imshow(self.get_PIL_image())
        plt.show()

    def get_PIL_image(self) -> Image.Image:
        if self.PIL_IMAGE is None:
            self.PIL_IMAGE = Image.new("RGB", self.dimensions)
            for x in range(self.width):
                for y in range(self.height):
                    self.PIL_IMAGE.putpixel((x, y), self[x, y])
        return self.PIL_IMAGE

    @staticmethod
    def show_images(images: list):
        if len(images) > 3:
            IMAGES_PER_ROW = 3
            NUMBER_OF_ROWS = (len(images) // IMAGES_PER_ROW + 1)
        else:
            IMAGES_PER_ROW = len(images)
            NUMBER_OF_ROWS = 1

        for x in range(IMAGES_PER_ROW):
            for y in range(NUMBER_OF_ROWS):
                i = x + y * IMAGES_PER_ROW
                if i >= len(images): continue
                img = images[i]
                axe = plt.subplot2grid(
                    (NUMBER_OF_ROWS, IMAGES_PER_ROW),
                    (y, x))

                pil_img = Image.new("RGB", img.dimensions)
                pil_img.putdata(list(zip(img.r.flatten(), img.g.flatten(), img.b.flatten()))) # type:ignore
                axe.imshow(pil_img)
        plt.show()

    @staticmethod
    def show_histograms(images: list, _type: Literal["h","nh","ch","cnh"]):
        HISTO_TYPES = "h", 'nh', "ch", "cnh"
        _type = _type.lower().strip() # type:ignore
        if _type not in HISTO_TYPES: raise ValueError(f"type of histogram {_type} is not supported please choose from {HISTO_TYPES}")

        def get_histo(img: MyImage, _type: str):
            match _type:
                case "h":
                    return img.histograme()
                case "nh":
                    return img.normalized_histograme()
                case "ch":
                    return img.cumulated_histograme()
                case "cnh":
                    return img.cumulative_normalized_histo()
                case _:
                    raise ValueError(f"type of histogram {_type} is not supported please choose from {HISTO_TYPES}")

        TITLES = {
            "h": "FREQUENCY HISTOGRAM",
            "nh": "NORMALIZED HISTOGRAM",
            "ch": "CUMULATED HISTOGRAM",
            "cnh": "C/N HISTOGRAM"
        }

        NUMBER_OF_ROWS = len(images)
        IMAGES_PER_ROW = 3

        for i, img in enumerate(images):
            if img.mode == "RGB":
                histo_r, histo_g, histo_b = get_histo(img, _type)

                axe = plt.subplot2grid(
                    (NUMBER_OF_ROWS, IMAGES_PER_ROW),
                    (i, 0))
                axe.plot(np.arange(256), histo_r, color='red')
                axe.set_title(TITLES.get(_type, "UNKNOWEN HISTOGRAM") + " RED CHANNEL")

                axe = plt.subplot2grid(
                    (NUMBER_OF_ROWS, IMAGES_PER_ROW),
                    (i, 1))
                axe.plot(np.arange(256), histo_g, color='green')
                axe.set_title(TITLES.get(_type, "UNKNOWEN HISTOGRAM") + " GREEN CHANNEL")

                axe = plt.subplot2grid(
                    (NUMBER_OF_ROWS, IMAGES_PER_ROW),
                    (i, 2))
                axe.plot(np.arange(256), histo_b, color='blue')
                axe.set_title(TITLES.get(_type, "UNKNOWEN HISTOGRAM") + " BLUE CHANNEL")

            elif img.mode == 'L':
                histo = get_histo(img, _type)
                axe = plt.subplot2grid(
                    (NUMBER_OF_ROWS, IMAGES_PER_ROW),
                    (i, 0), colspan=3)
                axe.plot(np.arange(256), histo, color='gray')
                axe.set_title(TITLES.get(_type, "UNKNOWEN HISTOGRAM") + " GRAY CHANNEL")

            else:
                raise Exception(f"{img.mode} is unsupported")

        plt.show()

