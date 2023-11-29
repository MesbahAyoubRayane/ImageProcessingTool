import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

class MyImage:
    """
    A class containging all the functions learned in the image processing class:
    support two types of images 'RGB' an 'L' (gray scale)
    """
    MODES = 'RGB','L'
    DEFAUL_GRAY_SCALE_COEF = (0.299,0.587,0.114) # this are the deafults values for transforming from rgb to gray scale

    def __init__(self,r:np.ndarray,g:np.ndarray,b:np.ndarray,mode:str):
        """
        r,g,b are numpy matrices the must have the same shape (width*height)
        mode : 'RGB' or 'L'
        """
        mode = mode.strip().upper()
        if mode not in MyImage.MODES: raise ValueError(f'Unsupported mode value {mode},mode must be L or RGB')
        if r.ndim != 2 or g.ndim != 2 or b.ndim != 2:
            raise Exception('R,G,B chanels must be in a matrix form')
        if not (r.shape == g.shape and r.shape == b.shape):
            raise Exception('The provided arrays are not cohierant')
        self.mode = mode
        self.r,self.g,self.b = r.copy().astype(np.uint8),g.copy().astype(np.uint8),b.copy().astype(np.uint8)

    @property
    def width(self): return len(self.r[0])
    
    @property
    def height(self): return len(self.r)

    @property
    def dimensions(self): return self.width,self.height

    def pixels(self):
        """
        this function return all the pixels of the image:
        if mode = rgb  ==> x,y,r,g,b
        if mode = l ==> x,y,v
        """
        if self.mode.upper() == 'RGB':
            for x in range(self.width):
                for y in range(self.height): yield x,y,self.r[y,x],self.g[y,x],self.b[y,x]
        
        elif self.mode.upper() == 'L':
            for x in range(self.width):
                for y in range(self.height): yield x,y,self.r[y,x]

    def __getitem__(self,indecies:(int,int)) -> tuple[int,int,int]:
        """
        used to index the image 
        P = img[x,y]
        P == (r,g,b) if mode = RGB
        P == (v,v,v) v is the gray scale value 
        """
        x,y = self.__test_indecies__(indecies[0],indecies[1])
        return int(self.r[y,x]),int(self.g[y,x]),int(self.b[y,x])
    
    def __setitem__(self,indecies:(int,int),value:tuple[int,int,int]|int):
        x,y = self.__test_indecies__(indecies[0],indecies[1])
        if self.mode.upper() == "RGB":
            r,g,b = int(value[0]),int(value[1]),int(value[2]) 
            for v in r,g,b:
                if not isinstance(v,int):
                    raise ValueError(f"the provided value is not an integer v={v}")
                if not 0<=v<256:
                    raise ValueError('RGB values must be between 0 and 255 inclusive')
            self.r[y,x],self.g[y,x],self.b[y,x] = r,g,b
        
        elif self.mode.upper() == 'L':
            if isinstance(value,tuple) or isinstance(value,list):
                value = value[0]
            value = int(value)
            if not 0<=value<256:
                raise ValueError('RGB values must be between 0 and 255 inclusive')
            self.r[y,x],self.g[y,x],self.b[y,x] = value,value,value

    def __test_indecies__(self,x,y):
        if not 0<=x<self.width:
            raise Exception(f"x value {x}, is greater than the width of the image {self.width}")
        if not 0<=y<self.height:
            raise Exception(f"y value {y}, is greater than the height of the image {self.height}")
        return int(x),int(y)

    def copy(self):
        # creat a deep copy of the image
        return MyImage(self.r,self.g,self.b,self.mode)

    def cut(self,x:int,y:int,w:int,h:int):
        """
        return a sub_image from the original image starting from the point x,y (top-left) to x+w,y+h (bottom-right)
        """
        x,y = self.__test_indecies__(x,y)

        w,h = int(w),int(h)
        if w <= 0 or h <= 0: raise ValueError(f"the width and height must be a positive value , w = {w},h = {h}")
        rimg =MyImage.new(w,h,self.mode)

        for xn in range(w):
            for yn in range(h):
                if x+xn < self.width and y+yn < self.height:
                    rimg[xn,yn] = self[xn+x,yn+y]
                
        return rimg

    # Geometric transformation
    def translation(self,vec:tuple[float|int,float|int]):
        """
        translate the image by a vector (x,y) -> (x+u,y+v) 
        """
        cpy = MyImage.new(self.width,self.height,self.mode)
        for x,y,*v in self.pixels():
            x_ = x + vec[0]
            y_ =y + vec[1]
            if 0<=x_<self.width and 0<= y_ < self.height:
                cpy[x_,y_] = v if self.mode == "RGB" else v[0]
        return cpy
    
    def paste(self,new_width:int,new_height:int,upper_left_point:tuple[int,int]):
        """
        This function allow you to paste another image on the new image, the new image must be of a larger size
        for it to contain the old image 
        """
        if self.width > new_width or self.height > new_height:
            raise ValueError('The image is bigger than the canvas')
        img = MyImage.new(new_width,new_height,self.mode)

        for x,y,*v in self.pixels():
            if x+upper_left_point[0] < img.width and y + upper_left_point[1] < img.height:
                img[x+upper_left_point[0],y+upper_left_point[1]] = v if self.mode == 'RGB' else v[0]
            
        return img  
    
    def lay(self,img):
        """
        take an image as an argument and lay the first image on the second resulting a new image containing both
        Note:
        In the implementation i'm doing a sum of each chanel , so the overlapping pixels will have a value of 255
        """
        img:MyImage = img
        if self.mode != img.mode: raise Exception("the images must be of the same mode")
        if img.dimensions != self.dimensions: raise Exception(f"You can't lay an image of size {img.dimensions} on an image of size {self.dimensions} , the size must be the same")
        cpy_img = MyImage.new(self.width,self.height,self.mode)

        cpy_img.r = (self.r.flatten().astype(np.uint32) + img.r.flatten().astype(np.uint32)).clip(0,255).astype(np.uint8).reshape(self.r.shape)
        cpy_img.g = (self.g.flatten().astype(np.uint32) + img.g.flatten().astype(np.uint32)).clip(0,255).astype(np.uint8).reshape(self.g.shape)
        cpy_img.b = (self.b.flatten().astype(np.uint32) + img.b.flatten().astype(np.uint32)).clip(0,255).astype(np.uint8).reshape(self.b.shape)

        return cpy_img

    def reflecte(self,axe:str):
        """
        mirrore the image on the horizantal or vertical axe
        axe : ['v','h']
        """
        axe = axe.lower()
        if axe not in ('h','v'):raise Exception("axe must be v or h")
        
        if axe == 'v':
            fimg = MyImage(
                np.array([row[::-1] for row in self.r]),
                np.array([row[::-1] for row in self.g]),
                np.array([row[::-1] for row in self.b]),self.mode)
        elif axe == 'h':
            fimg = MyImage(np.array([row for row in self.r[::-1]]),np.array([row for row in self.g[::-1]]),np.array([row for row in self.b[::-1]]),self.mode)

        return fimg
    
    def rotate(self,theta:float):
        rotated_img = MyImage.new(self.width,self.height,self.mode)
        W,H = rotated_img.width,rotated_img.height
        theta = theta * np.pi/180
        theta *= -1
        COS ,SIN = np.cos(theta),np.sin(theta)
        rotation_matrix_t = np.array([
            [COS,-SIN],
            [SIN,COS]]
        ).transpose()

        U_V = np.array([[i,j] for i in range(W) for j in range(H)]).transpose()
        U_V_MINUS_CENTER = U_V - np.array([[W//2]*H*W,[H//2]*H*W])
        X_Y = (rotation_matrix_t @ U_V_MINUS_CENTER) + np.array([[W//2]*H*W,[H//2]*H*W])
        
        U_V = U_V.transpose()
        X_Y = X_Y.transpose()

        for i in range(W*H):
            u,v = U_V[i].tolist()
            x,y = X_Y[i].tolist()
            try:
                rotated_img[u,v] = self[x,y]
            except Exception:
                continue
        return rotated_img 

    def rescale(self,x_scaling_factor:float,y_scaling_factor:float):
        if x_scaling_factor <= 0 or y_scaling_factor <= 0:
            raise ValueError("The selected factors are incorrect")
        
        NW = int(self.width * x_scaling_factor)
        NH = int(self.height * y_scaling_factor)

        scaled_img = MyImage.new(NW,NH,self.mode)

        for x in range(NW):
            for y in range(NH):
                try:
                    scaled_img[x,y] = self[int(x/x_scaling_factor),int(y/y_scaling_factor)]
                except Exception:
                    continue
        return scaled_img
    
    def resolution_underscaling(self,factor:int):
        """
        this function divide the range of each chanel into X bages and affect the mean of each bag to the colors laying inside the range
        exemple:
        factor = 32
        [0:32] [32:64] ... [224:256]
        each pixel laying between 0 and 32 will have the value (0+32)/2
        """
        # continfication
        factor = int(factor)
        if not (0<factor<256): raise ValueError(f'the factor must bet 0<factor<256 but factor = {factor}')
        if 256 % factor != 0 : raise ValueError(f"256 must be divisibale by fcator but 256 % {factor} != 0")
        
        img = MyImage.new(self.width,self.height,self.mode)
        backets = {i//factor:(i+factor + i)//2 for i in range(0,256,factor)}
        def f(x): return backets[x]
        f = np.vectorize(f)
        
        img.r =f((self.r.flatten() / factor).astype(np.uint32)).astype(np.uint8).reshape(self.r.shape)
        img.g =f((self.g.flatten() / factor).astype(np.uint32)).astype(np.uint8).reshape(self.g.shape)
        img.b =f((self.b.flatten() / factor).astype(np.uint32)).astype(np.uint8).reshape(self.b.shape)

        return img

    # histogram based operations
    def histo_translation(self,t:int):
        nr = np.clip(self.r.astype(np.int32)+t,0,255).astype(np.uint8)
        ng = np.clip(self.g.astype(np.int32)+t,0,255).astype(np.uint8)
        nb = np.clip(self.b.astype(np.int32)+t,0,255).astype(np.uint8)
        return MyImage(nr,ng,nb,self.mode)
    
    def histo_inverse(self):
        r = 255 - np.array(self.r)
        g = 255 - np.array(self.g)
        b = 255 - np.array(self.b)
        return MyImage(r,g,b,self.mode) 
    
    def histo_expansion_dynamique(self):
        """
        Note before using this function you need to remove outliers because then can change the results dramaticly
        This function is just a simple normalization function between 0 and 255 , thus outliers have an important effect on the function
        """
        if self.mode == "RGB":
            MIN = self.r.flatten().min()
            MAX = self.r.flatten().max()
            r = np.array((self.r.flatten().astype(np.float64) - MIN )* (255/(MAX-MIN)),dtype=np.uint8).reshape(self.r.shape)
            MIN = self.g.flatten().min()
            MAX = self.g.flatten().max()
            g = np.array((self.g.flatten().astype(np.float64) - MIN )* (255/(MAX-MIN)),dtype=np.uint8).reshape(self.g.shape)
            MIN = self.b.flatten().min()
            MAX = self.b.flatten().max()
            b = np.array((self.b.flatten().astype(np.float64) - MIN )* (255/(MAX-MIN)),dtype=np.uint8).reshape(self.b.shape)
            return MyImage(r,g,b,self.mode)
        elif self.mode == 'L':
            MIN = self.r.flatten().min()
            MAX = self.r.flatten().max()
            gray = np.array((self.r.flatten().astype(np.float64) - MIN )* (255/(MAX-MIN)),dtype=np.uint8).reshape(self.r.shape)
            return MyImage(gray,gray,gray,self.mode)

    def histo_equalisation(self):
        """ use the cumulative histograme to improve contraste"""
        if self.mode == "RGB":
            cdf_r,cdf_g,cdf_b = self.cumulative_normilized_histo()
            cp_img = self.copy()
            for x,y,r,g,b in self.pixels():
                cp_img[x,y] = (int(cdf_r[r]*255),int(cdf_g[g]*255),int(cdf_b[b]*255))
        elif self.mode == 'L':
            cdf:np.ndarray = self.cumulative_normilized_histo()
            cp_img = self.copy()
            for x,y,v in cp_img.pixels():
                cp_img[x,y] = int(255 * cdf[int(v)])
        return cp_img

    def histo_matching(self,model):
        """use an image as a model for another image"""
        if isinstance(model,MyImage):
            if self.mode != model.mode:
                raise ValueError("The selected image model doesn't have the samel mode as the modeled image")
            cpyimg = self.copy()
            if self.mode == "L":
                cnh_model:np.ndarray = model.cumulative_normilized_histo()
                for x,y,v in self.pixels():
                    cpyimg[x,y] = int(255 * cnh_model[v])
        
            elif self.mode == "RGB":
                cnh_model_r,cnh_model_g,cnh_model_b = model.cumulative_normilized_histo()
                for x,y,r,g,b in self.pixels():
                    cpyimg[x,y] = (int(255*cnh_model_r[r]),int(cnh_model_g[g]),int(cnh_model_b[b]))
        elif isinstance(model,np.ndarray):
            cpyimg = self.copy()
            if self.mode == "L":
                cnh_model = model
                for x,y,v in self.pixels():
                    cpyimg[x,y] = int(255 * cnh_model[v])
        
            elif self.mode == "RGB":
                cnh_model_r,cnh_model_g,cnh_model_b = model.cumulative_normilized_histo()
                for x,y,r,g,b in self.pixels():
                    cpyimg[x,y] = (int(255*cnh_model_r[r]),int(cnh_model_g[g]),int(cnh_model_b[b]))
 
        return cpyimg    

    # filters
    def gray_scale(self):
        coef = MyImage.DEFAUL_GRAY_SCALE_COEF
        Gray = np.array((self.r * coef[0] + self.g * coef[1] + self.b * coef[2]) / sum(coef),dtype=np.uint8)
        return MyImage(Gray,Gray,Gray,'L')

    def mean_filter(self,size:int):
        if isinstance(size,int):
            if size < 2:
                raise ValueError(f'size must be > 2')
            if size %2 == 0:
                raise ValueError(f"The size must be odd number")
            if size > self.width or size >self.height:
                raise ValueError(f'the provided size is so large')
        else:
            raise ValueError(f"{type(size)} can't be used as a filter")
        
        copy_img = self.copy()
        kernel = np.full((1,size,size),1/(size**2))

        r_pad = np.pad(self.r,((size//2,size//2),(size//2,size//2)),mode='reflect')
        g_pad = np.pad(self.g,((size//2,size//2),(size//2,size//2)),mode='reflect')
        b_pad = np.pad(self.b,((size//2,size//2),(size//2,size//2)),mode='reflect')

        r_bag = np.array(
            [r_pad[y - size //2 :y + size//2 + 1, x - size//2: x + size//2 +1]
            for y in range(size//2,self.height+size//2)
            for x in range(size//2,self.width+size//2)]
        )
        g_bag = np.array(
            [g_pad[y - size //2 :y + size//2 + 1, x - size//2: x + size//2 +1]
            for y in range(size//2,self.height+size//2)
            for x in range(size//2,self.width+size//2)]
        )
        b_bag = np.array(
            [b_pad[y - size //2 :y + size//2 + 1, x - size//2: x + size//2 +1]
            for y in range(size//2,self.height+size//2)
            for x in range(size//2,self.width+size//2)]
        )

        copy_img.r = np.clip((r_bag * kernel).sum(axis=(1,2)),0,255).astype(np.uint8).reshape(self.r.shape)
        copy_img.g = np.clip((g_bag * kernel).sum(axis=(1,2)),0,255).astype(np.uint8).reshape(self.r.shape)
        copy_img.b = np.clip((b_bag * kernel).sum(axis=(1,2)),0,255).astype(np.uint8).reshape(self.r.shape)
        return copy_img

    # TODO this function can be improved using two convolution the first on the x axis and the second on the y axes
    def gaussian_filter(self, size: int, std: float):
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
        kernel /= kernel.sum()
        # Normalize the kernel
        #kernel /= kernel.sum() this is not necessary 
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

            copy_img = MyImage.new(self.width,self.height,self.mode)

            all_r_patchs = np.array(
                [extended_r[y - size //2 :y + size//2 + 1, x - size//2: x + size//2 +1]
                for y in range(size//2,self.height+size//2)
                for x in range(size//2,self.width + size//2)
                ]
            )

            all_g_patchs = np.array(
                [extended_g[y - size // 2:y + size // 2 + 1, x - size // 2:x + size // 2 + 1] 
                for y in range(size//2,self.height+size//2)
                for x in range(size//2,self.width + size//2)
                ]
            )

            all_b_patchs = np.array(
                [extended_b[y - size // 2:y + size // 2 + 1, x - size // 2:x + size // 2 + 1] 
                for y in range(size//2,self.height+size//2)
                for x in range(size//2,self.width + size//2)
                ]
            )

            kernel = kernel.reshape((1,size,size))
            all_r_conv:np.ndarray = np.clip((all_r_patchs * kernel).sum(axis=(1,2)),0,255).reshape(self.r.shape)
            all_g_conv:np.ndarray = np.clip((all_g_patchs * kernel).sum(axis=(1,2)),0,255).reshape(self.r.shape)
            all_b_conv:np.ndarray = np.clip((all_b_patchs * kernel).sum(axis=(1,2)),0,255).reshape(self.r.shape)


            copy_img.r = all_r_conv.astype(np.uint8)
            copy_img.g = all_g_conv.astype(np.uint8)
            copy_img.b = all_b_conv.astype(np.uint8)

            return copy_img
        
        elif self.mode == "L":
            extended_r = np.pad(self.r, ((size // 2, size // 2), (size // 2, size // 2)), 'reflect')
            copy_img = MyImage.new(self.width,self.height,self.mode)
            all_r_patchs = np.array(
                [extended_r[y - size //2 :y + size//2 + 1, x - size//2: x + size//2 +1]
                for y in range(size//2,self.height+size//2)
                for x in range(size//2,self.width + size//2)
                ]
            )
            kernel = kernel.reshape((1,size,size))
            all_r_conv:np.ndarray = np.clip((all_r_patchs * kernel).sum(axis=(1,2)),0,255).reshape(self.r.shape)
            copy_img.r = all_r_conv.astype(np.uint8)
            copy_img.g = copy_img.r
            copy_img.b = copy_img.g
            return copy_img
        else:
            raise ValueError(f"{self.mode} is not supported")

    # TODO if i have time this is a good filter to implement
    def bilateral_filter(self,size:int,std_spatial_gaussian:float,std_brightness_gaussian:float):
        size = int(size)
        std_s = float(std_spatial_gaussian)
        std_b = float(std_brightness_gaussian)
        
        if size < 2:
            raise ValueError(f'size must be > 2')
        if size %2 == 0:
            raise ValueError(f"The size must be odd number")
        if size > self.width or size >self.height:
            raise ValueError(f'the provided size is so large')
        
        if std_b <= 0 or std_s <= 0:
            raise ValueError(f"std value must be > 0 {std_s,std_b}")
        
        X,Y = np.meshgrid(np.arange(size),np.arange(size))
        s_kernel =  (np.exp(-0.5 * ((X - size//2) ** 2 + (Y - size //2) ** 2)/std_s ** 2) / (2*np.pi*std_s**2)).reshape((1,size,size))
        cpy_img = MyImage.new(self.width,self.height,self.mode)

        if self.mode == "RGB":
            extended_r = np.pad(self.r, pad_width=size//2 , mode='reflect')
            extended_g = np.pad(self.g, pad_width=size//2 , mode='reflect')
            extended_b = np.pad(self.b, pad_width=size//2 , mode='reflect')

            all_r_patchs = np.array(
                [extended_r[y - size //2 :y + size//2 + 1, x - size//2: x + size//2 +1]
                for y in range(size//2,self.height+size//2)
                for x in range(size//2,self.width + size//2)
                ]
            )

            all_g_patchs = np.array(
                [extended_g[y - size // 2:y + size // 2 + 1, x - size // 2:x + size // 2 + 1] 
                for y in range(size//2,self.height+size//2)
                for x in range(size//2,self.width + size//2)
                ]
            )

            all_b_patchs = np.array(
                [extended_b[y - size // 2:y + size // 2 + 1, x - size // 2:x + size // 2 + 1] 
                for y in range(size//2,self.height+size//2)
                for x in range(size//2,self.width + size//2)
                ]
            )

            # gaussian kernels for red
            b_r_kernel = np.array(
                [
                    (extended_r[y - size //2 :y + size//2 + 1, x - size//2: x + size//2 +1] - 
                    np.full((size,size),extended_r[y,x])) ** 2
                    for y in range(size//2,self.height+size//2)
                    for x in range(size//2,self.width + size//2)
                ]
            )
            b_r_kernel = np.exp(b_r_kernel / np.full((1,size,size),-2*std_b**2)) / (np.sqrt(2*np.pi)*std_b)
        
            # gaussian kernels for green
            b_g_kernel = np.array(
                [
                    (extended_g[y - size //2 :y + size//2 + 1, x - size//2: x + size//2 +1] - 
                    np.full((size,size),extended_g[y,x])) ** 2
                    for y in range(size//2,self.height + size//2)
                    for x in range(size//2,self.width + size//2)
                ]
            ) 
            b_g_kernel = np.exp(b_g_kernel / np.full((1,size,size),-2*std_b**2)) / (np.sqrt(2*np.pi)*std_b)

            
            # gaussian kernels for blue
            b_b_kernel = np.array(
                [
                    (extended_b[y - size //2 :y + size//2 + 1, x - size//2: x + size//2 +1] - 
                    np.full((size,size),extended_b[y,x])) ** 2
                    for y in range(size//2,self.height + size//2)
                    for x in range(size//2,self.width + size//2)
                ]
            ) 
            b_b_kernel = np.exp(b_b_kernel / np.full((1,size,size),-2*std_b**2)) / (np.sqrt(2*np.pi)*std_b)

             
            # compute the new values of the pixels
            tmp =  s_kernel * b_r_kernel
            new_r = (all_r_patchs * tmp).sum(axis=(1,2)) /tmp.sum(axis=(1,2))
            tmp =  s_kernel * b_g_kernel
            new_g = (all_g_patchs * tmp).sum(axis=(1,2)) /tmp.sum(axis=(1,2))
            tmp = s_kernel * b_b_kernel
            new_b = (all_b_patchs * tmp).sum(axis=(1,2)) /tmp.sum(axis=(1,2))
            

            cpy_img.r = np.clip(new_r,0,255).astype(np.uint8).reshape(self.r.shape)
            cpy_img.g = np.clip(new_g,0,255).astype(np.uint8).reshape(self.r.shape)
            cpy_img.b = np.clip(new_b,0,255).astype(np.uint8).reshape(self.r.shape)

            return cpy_img
        
        elif self.mode == 'L':
            extended_r = np.pad(self.r, pad_width=size//2 , mode='reflect')
            all_r_patchs = np.array(
                [extended_r[y - size //2 :y + size//2 + 1, x - size//2: x + size//2 +1]
                for y in range(size//2,self.height+size//2)
                for x in range(size//2,self.width + size//2)
                ]
            )
            b_r_kernel = np.array(
                [
                    (extended_r[y - size //2 :y + size//2 + 1, x - size//2: x + size//2 +1] - 
                    np.full((size,size),extended_r[y,x])) ** 2
                    for y in range(size//2,self.height+size//2)
                    for x in range(size//2,self.width + size//2)
                ]
            )
            tmp =  s_kernel * b_r_kernel
            new_r = (all_r_patchs * tmp).sum(axis=(1,2)) /tmp.sum(axis=(1,2))
            cpy_img.b = cpy_img.g = cpy_img.r = np.clip(new_r,0,255).astype(np.uint8).reshape(self.r.shape)
            return cpy_img
        else:
            raise ValueError(f"{self.mode} is not supported")

    def median_filter(self,size:int):
        if isinstance(size,int):
            if size < 2:
                raise ValueError(f'size must be > 2')
            if size %2 == 0:
                raise ValueError(f"The size must be odd number")
            if size > self.width or size >self.height:
                raise ValueError(f'the provided size is so large')
        else:
            raise ValueError(f"{type(size)} can't be used as a filter")
        
        cpy_img = MyImage.new(self.width,self.height,self.mode)
        
        if self.mode == "RGB":
            pad_r = np.pad(self.r,size//2,"reflect")
            pad_g = np.pad(self.g,size//2,"reflect")
            pad_b = np.pad(self.b,size//2,"reflect")

            r_bag = np.array(
                [pad_r[y - size //2 :y + size//2 + 1, x - size//2: x + size//2 +1]
                for y in range(size//2,self.height+size//2)
                for x in range(size//2,self.width + size//2)
                ]
            )
            g_bag = np.array(
                [pad_g[y - size //2 :y + size//2 + 1, x - size//2: x + size//2 +1]
                for y in range(size//2,self.height+size//2)
                for x in range(size//2,self.width + size//2)
                ]
            )
            b_bag = np.array([
                pad_b[y - size //2 :y + size//2 + 1, x - size//2: x + size//2 +1]
                for y in range(size//2,self.height+size//2)
                for x in range(size//2,self.width + size//2)
                ]
            )

            cpy_img.r = np.median(r_bag,axis=(1,2)).reshape(self.r.shape).astype(np.uint8)
            cpy_img.g = np.median(g_bag,axis=(1,2)).reshape(self.r.shape).astype(np.uint8)
            cpy_img.b = np.median(b_bag,axis=(1,2)).reshape(self.r.shape).astype(np.uint8)

            return cpy_img

        elif self.mode == 'L':
            pad_r = np.pad(self.r,size//2,"reflect")
            r_bag = np.array(
                [pad_r[y - size //2 :y + size//2 + 1, x - size//2: x + size//2 +1]
                for y in range(size//2,self.height+size//2)
                for x in range(size//2,self.width + size//2)
                ]
            )
            cpy_img.r = np.median(r_bag,axis=(1,2)).reshape(self.r.shape).astype(np.uint8)
            cpy_img.g = cpy_img.b = cpy_img.r
            return cpy_img
        else:
            raise ValueError(f"{self.mode} is not supported")
    
    def laplacian_sharpning_filter(self,distance:str,size:int):
        """
        distance must be on of these variants : manhatten,max
        size : is an odd positive number
        """
        size = int(size)
        if size < 2:raise ValueError('size must be > 2')
        if size % 2 == 0:raise ValueError('size must be odd number')

        distance = distance.lower().strip()
        if distance == 'manhatten':
            kernel = np.zeros((size,size))
            kernel[size//2,:] = 1
            kernel[:,size//2] = 1
            kernel[size//2,size//2] = -(kernel.sum() - 1) 
        elif distance == "max":
            kernel = np.ones((size,size))
            kernel[size//2,size//2] = -(kernel.sum() - 1) 
        else:
            raise ValueError("distance must be 4 or 8")
        kernel = kernel.reshape((1,size,size))

        r_pad = np.pad(self.r,size//2,'reflect')
        g_pad = np.pad(self.g,size//2,'reflect')
        b_pad = np.pad(self.b,size//2,'reflect')
        copy_img = MyImage.new(self.width,self.height,self.mode)
        r_bag = np.array(
            [r_pad[y - size //2 :y + size//2 + 1, x - size//2: x + size//2 +1]
            for y in range(size//2,self.height+size//2)
            for x in range(size//2,self.width+size//2)]
        )
        g_bag = np.array(
            [g_pad[y - size //2 :y + size//2 + 1, x - size//2: x + size//2 +1]
            for y in range(size//2,self.height+size//2)
            for x in range(size//2,self.width+size//2)]
        )
        b_bag = np.array(
            [b_pad[y - size //2 :y + size//2 + 1, x - size//2: x + size//2 +1]
            for y in range(size//2,self.height+size//2)
            for x in range(size//2,self.width+size//2)]
        )

        copy_img.r = np.clip((r_bag * kernel).sum(axis=(1,2)),0,255).astype(np.uint8).reshape(self.r.shape)
        copy_img.g = np.clip((g_bag * kernel).sum(axis=(1,2)),0,255).astype(np.uint8).reshape(self.r.shape)
        copy_img.b = np.clip((b_bag * kernel).sum(axis=(1,2)),0,255).astype(np.uint8).reshape(self.r.shape)

        return copy_img
    
    def edge_detection_robert(self,threshold:int):
        kernel_diag = np.array([[-1,0],[0,1]]).reshape((1,2,2))
        kernel_rev_diag = np.array([[0,-1],[1,0]]).reshape((1,2,2))

        extended_r = np.pad(self.r,pad_width=1,mode='reflect')
        extended_g = np.pad(self.g,pad_width=1,mode='reflect')
        extended_b = np.pad(self.b,pad_width=1,mode='reflect')

        W,H = self.width,self.height

        bage_r = np.array([
            extended_r[y:y+2,x:x+2]
            for y in range(1,H+1)
            for x in range(1,W+1)
        ])
        bage_g = np.array([
            extended_g[y:y+2,x:x+2]
            for y in range(1,H+1)
            for x in range(1,W+1)
        ])
        bage_b = np.array([
            extended_b[y:y+2,x:x+2]
            for y in range(1,H+1)
            for x in range(1,W+1)
        ])
        G_diag_r = (bage_r * kernel_diag).sum(axis=(1,2))
        G_diag_g = (bage_g * kernel_diag).sum(axis=(1,2))
        G_diag_b = (bage_b * kernel_diag).sum(axis=(1,2))

        G_rev_diag_r = (bage_r * kernel_rev_diag).sum(axis=(1,2))
        G_rev_diag_g = (bage_g * kernel_rev_diag).sum(axis=(1,2))
        G_rev_diag_b = (bage_b * kernel_rev_diag).sum(axis=(1,2))

        G_r:np.ndarray = np.sqrt(G_diag_r ** 2 + G_rev_diag_r ** 2)
        G_g:np.ndarray = np.sqrt(G_diag_g ** 2 + G_rev_diag_g ** 2)
        G_b:np.ndarray = np.sqrt(G_diag_b ** 2 + G_rev_diag_b ** 2)

        G_r:np.ndarray = (G_r - G_r.min())/(G_r.max() - G_r.min()) * 255
        G_g:np.ndarray = (G_g - G_g.min())/(G_g.max() - G_g.min()) * 255
        G_b:np.ndarray = (G_b - G_b.min())/(G_b.max() - G_b.min()) * 255

        f = np.vectorize(lambda x:255 if x > threshold else 0)
        G_r = f(G_r)
        G_g = f(G_g)
        G_b = f(G_b)

        G_r = G_r.astype(np.uint8).reshape(self.r.shape)
        G_g = G_g.astype(np.uint8).reshape(self.r.shape)
        G_b = G_b.astype(np.uint8).reshape(self.r.shape)

        return MyImage(G_r,G_g,G_b,self.mode)
    
    def edge_detection_sobel(self,threshold:int):
        """
        threshold is an integer value between 0 and 255 , the lower it is the more daitaills will be detected as an edge
        """
        kernel_h = np.array([
            [-1,0,1],
            [-2,0,2],
            [-1,0,1]
        ]).reshape((1,3,3))
        kernel_v = np.array([
            [-1,-2,-1],
            [0,0,0],
            [1,2,1]
        ]).reshape((1,3,3))

        W,H = self.width,self.height

        extended_r = np.pad(self.r,1,'reflect')
        extended_g = np.pad(self.g,1,'reflect')
        extended_b = np.pad(self.b,1,'reflect')
        # creating the bags
        bage_r = np.array([
            extended_r[y-1:y+2,x-1:x+2]
            for y in range(1,H+1)
            for x in range(1,W+1)
        ])
        bage_g = np.array([
            extended_g[y-1:y+2,x-1:x+2]
            for y in range(1,H+1)
            for x in range(1,W+1)
        ])
        bage_b = np.array([
            extended_b[y-1:y+2,x-1:x+2]
            for y in range(1,H+1)
            for x in range(1,W+1)
        ])
        G_r_h = (bage_r * kernel_h).sum(axis=(1,2))
        G_r_v = (bage_r * kernel_v).sum(axis=(1,2))
        G_g_h = (bage_g * kernel_h).sum(axis=(1,2))
        G_g_v = (bage_g * kernel_v).sum(axis=(1,2))
        G_b_h = (bage_b * kernel_h).sum(axis=(1,2))
        G_b_v = (bage_b * kernel_v).sum(axis=(1,2))

        G_r = np.sqrt(G_r_h ** 2 + G_r_v ** 2)
        G_g = np.sqrt(G_g_h ** 2 + G_g_v ** 2)
        G_b = np.sqrt(G_b_h ** 2 + G_b_v ** 2)

        # normilizing the gradiants from 0..255
        G_r = (G_r - G_r.min())/(G_r.max() - G_r.min()) * 255
        G_g = (G_g - G_g.min())/(G_g.max() - G_g.min()) * 255
        G_b = (G_b - G_b.min())/(G_b.max() - G_b.min()) * 255
        
        f = np.vectorize(lambda x:255 if x > threshold else 0)
        G_r:np.ndarray = f(G_r)
        G_g:np.ndarray = f(G_g)
        G_b:np.ndarray = f(G_b)

        G_r = G_r.astype(np.uint8).reshape(self.r.shape)
        G_g = G_g.astype(np.uint8).reshape(self.r.shape)
        G_b = G_b.astype(np.uint8).reshape(self.r.shape)

        return MyImage(G_r,G_g,G_b,self.mode)

    def edges_detection_prewitt(self,threshold:int):
        """
        threshold is an integer value between 0 and 255 , the lower it is the more daitaills will be detected as an edge
        """
        kernel_h = np.array([
            [-1,0,1],
            [-1,0,1],
            [-1,0,1]
        ]).reshape((1,3,3))
        kernel_v = np.array([
            [-1,-1,-1],
            [0,0,0],
            [1,1,1]
        ]).reshape((1,3,3))

        W,H = self.width,self.height

        extended_r = np.pad(self.r,1,'reflect')
        extended_g = np.pad(self.g,1,'reflect')
        extended_b = np.pad(self.b,1,'reflect')
        # creating the bags
        bage_r = np.array([
            extended_r[y-1:y+2,x-1:x+2]
            for y in range(1,H+1)
            for x in range(1,W+1)
        ])
        bage_g = np.array([
            extended_g[y-1:y+2,x-1:x+2]
            for y in range(1,H+1)
            for x in range(1,W+1)
        ])
        bage_b = np.array([
            extended_b[y-1:y+2,x-1:x+2]
            for y in range(1,H+1)
            for x in range(1,W+1)
        ])
        G_r_h = (bage_r * kernel_h).sum(axis=(1,2))
        G_r_v = (bage_r * kernel_v).sum(axis=(1,2))
        G_g_h = (bage_g * kernel_h).sum(axis=(1,2))
        G_g_v = (bage_g * kernel_v).sum(axis=(1,2))
        G_b_h = (bage_b * kernel_h).sum(axis=(1,2))
        G_b_v = (bage_b * kernel_v).sum(axis=(1,2))

        G_r = np.sqrt(G_r_h ** 2 + G_r_v ** 2)
        G_g = np.sqrt(G_g_h ** 2 + G_g_v ** 2)
        G_b = np.sqrt(G_b_h ** 2 + G_b_v ** 2)

        # normilizing the gradiants from 0..255
        G_r = (G_r - G_r.min())/(G_r.max() - G_r.min()) * 255
        G_g = (G_g - G_g.min())/(G_g.max() - G_g.min()) * 255
        G_b = (G_b - G_b.min())/(G_b.max() - G_b.min()) * 255
        
        f = np.vectorize(lambda x:255 if x > threshold else 0)
        G_r:np.ndarray = f(G_r)
        G_g:np.ndarray = f(G_g)
        G_b:np.ndarray = f(G_b)

        G_r = G_r.astype(np.uint8).reshape(self.r.shape)
        G_g = G_g.astype(np.uint8).reshape(self.r.shape)
        G_b = G_b.astype(np.uint8).reshape(self.r.shape)

        return MyImage(G_r,G_g,G_b,self.mode)
    # clustering algorithms
    def kmean(self,k:int):
        def mean_rgb(v:list[tuple[int,int]]):
            r = g = b = 0
            N = len(v)
            for p in v:
                x,y = p
                r += self.r[y,x]
                b += self.b[y,x]
                g += self.g[y,x]
            
            r /= N
            g /= N
            b /= N
            return (r,g,b)

        def mean_l(v:list[tuple[int,int]]):
            g = 0
            N = len(v)
            for p in v:
                x,y = p
                g += self.r[y,x]
            return g/N

        k = int(k)
        if k <= 1: raise ValueError("k must be > 1")

        if self.mode == "RGB":
            clusters:dict[tuple[int,int,int],list[tuple[int,int]]] = {}
            while len(clusters) < k:
                x = np.random.randint(0,self.width)
                y = np.random.randint(0,self.height)
                r,g,b = self[x,y]
                if (x,y) in clusters:continue
                clusters[(r,g,b)] = [(x,y)]
            
            distances_matrix = np.full((k,self.width * self.height),0) 
            jumped = True
            while jumped:
                jumped = False
                for i,clus in enumerate(clusters.keys()):
                    rc,gc,bc = clus
                    distances_matrix[i] = np.sqrt(
                        (self.r.flatten() - rc)**2 + (self.g.flatten() - gc)**2 + (self.b.flatten() - bc)**2)
                
                
                new_position = distances_matrix.argsort(axis=0)[0]
                correspondance = {i:k for i,k in enumerate(clusters.keys())}
                new_clusters = {k:[] for k in clusters.keys()}
                for i in range(self.width*self.height):
                    c = correspondance[int(new_position[i])]
                    x,y = int(i%self.width),i//self.width
                    new_clusters[c].append((x,y))
                
                for k in clusters.keys():
                    if clusters[k] != new_clusters[k]:
                        jumped = True
                        break
                clusters = {}
                for k,v in new_clusters.items():
                    clusters[mean_rgb(v)] = v

        elif self.mode == 'L':
            clusters:dict[tuple[int|float],list[tuple[int,int]]] = {}
            
            while len(clusters) < k:
                x = np.random.randint(0,self.width)
                y = np.random.randint(0,self.height)
                g,_,_ = self[x,y]
                if (x,y) in clusters:continue
                clusters[g] = [(x,y)]
            
            distances_matrix = np.full((k,self.width * self.height),0) 
            jumped = True
            while jumped:
                jumped = False
                for i,clus in enumerate(clusters.keys()):
                    gc = clus
                    distances_matrix[i] = np.abs(self.r.flatten() - gc)
                
                
                new_position = distances_matrix.argsort(axis=0)[0]
                correspondance = {i:k for i,k in enumerate(clusters.keys())}
                new_clusters = {k:[] for k in clusters.keys()}
                for i in range(self.width*self.height):
                    c = correspondance[int(new_position[i])]
                    x,y = int(i%self.width),i//self.width
                    new_clusters[c].append((x,y))
                
                for k in clusters.keys():
                    if clusters[k] != new_clusters[k]:
                        jumped = True
                        break
                clusters = {}
                for k,v in new_clusters.items():
                    clusters[mean_l(v)] = v
        else:
            raise ValueError(f"{self.mode} is not suppotred")
        
        sub_imgs:list[MyImage] = []
        if self.mode == "RGB":
            for k,v in clusters.items():
                img = MyImage.new(self.width,self.height,self.mode)
                while k == (0,0,0):
                    k = np.random.randint(0,256),np.random.randint(0,256),np.random.randint(0,256)
                
                for p in v:
                    img[p] = k
                
                sub_imgs.append(img)
        
        elif self.mode =='L':
            for k,v in clusters.items():
                img = MyImage.new(self.width,self.height,self.mode)
                while k == 0:
                    k = np.random.randint(0,256)
                for p in v:
                    img[p] = k
                
                sub_imgs.append(img)
        return sub_imgs
    
    def binary_tagging(self):
        def get_neighbores(x:int,y:int): return [(i,j) for i in (x,x+1,x-1) for j in (y,y-1,y+1)]
        m:np.ndarray =np.zeros(self.r.shape)
        colored:np.ndarray = (self.r > 0)|(self.g > 0)|(self.b > 0)
        Tag = 1
        for x in range(self.width):
            for y in range(self.height):
                if m[y,x] != 0 or colored[y,x] == False:continue
                m[y,x] = Tag
                neighbores = get_neighbores(x,y)
                while len(neighbores) > 0:
                    xn,yn = neighbores.pop()
                    if not (0<=xn<self.width and 0<=yn<self.height): continue 
                    if m[yn,xn] != 0 or colored[yn,xn] == False:continue
                    m[yn,xn] = Tag
                    neighbores.extend(get_neighbores(xn,yn)) 
                Tag += 1
        colors = set()
        while len(colors) < Tag:
            colors.add((np.random.randint(0,256,dtype=np.uint8),
                   np.random.randint(0,256,dtype=np.uint8),
                   np.random.randint(0,256,dtype=np.uint8)))
        colors = list(colors)
        img = MyImage.new(self.width,self.height,self.mode)
        for x in range(self.width):
            for y in range(self.height):
                if m[y,x] != 0:
                    r,g,b = colors[int(m[y,x])]
                    img.r[y,x] = r
                    img.g[y,x] = g
                    img.b[y,x] = b
        
        return img     
    
    # histogrames
    def histograme(self) -> np.ndarray|tuple[np.ndarray,np.ndarray,np.ndarray]:
        if self.mode == "L":
            h = np.full((256,),fill_value=0)
            for v in self.r.flatten():
                h[v] += 1
            return h
        
        elif self.mode == 'RGB':
            hr,hg,hb = np.full((256,),fill_value=0),np.full((256,),fill_value=0),np.full((256,),fill_value=0) 
            for _,_,r,g,b in self.pixels():
                hr[r] += 1
                hg[g] += 1
                hb[b] += 1
            return hr,hg,hb
    
    def cumulated_histograme(self) ->  np.ndarray|tuple[np.ndarray,np.ndarray,np.ndarray]:
        if self.mode == "RGB":
            hr,hg,hb = self.histograme()
            chr,chg,chb = np.full((256,),dtype=np.int32,fill_value=0),np.full((256,),dtype=np.int32,fill_value=0),np.full((256,),dtype=np.int32,fill_value=0)
            assert hr.sum() == self.width * self.height
            assert hg.sum() == self.width * self.height
            assert hb.sum() == self.width * self.height
            sum_r = sum_g = sum_b = 0
            for i in range(256):
                sum_r += hr[i]
                sum_g += hg[i]
                sum_b += hb[i]
                chr[i] = sum_r
                chg[i] = sum_g
                chb[i] = sum_b
            return chr,chg,chb
        elif self.mode == 'L':
            hgray = self.histograme()
            chgray = np.full((256,),dtype=np.int32,fill_value=0)
            sum_gray = 0
            for i in range(256):
                sum_gray += hgray[i]
                chgray[i] = sum_gray
            return chgray

    def normilized_histograme(self) -> np.ndarray|tuple[np.ndarray,np.ndarray,np.ndarray]:
        if self.mode == "RGB":
            hr,hg,hb = self.histograme()
            nhr,nhg,nhb = np.array(hr,dtype=np.float64),np.array(hg,dtype=np.float64),np.array(hb,dtype=np.float64) 
            nhr /= (self.width*self.height)
            nhg /= (self.width*self.height)
            nhb /= (self.width*self.height)
            return  nhr,nhg,nhb
        elif self.mode == 'L':
            hgray = self.histograme()
            nhgray = np.array(hgray,dtype=np.float64) / (self.width*self.height)
            return nhgray
    
    def cumulative_normilized_histo(self) -> np.ndarray|tuple[np.ndarray,np.ndarray,np.ndarray]:
        if self.mode == 'RGB':
            nhr,nhg,nhb = self.normilized_histograme()
            cnhr,cnhg,cnhb = np.full(256,0.),np.full(256,0.),np.full(256,0.)
            cnhr[0] = nhr[0]
            cnhg[0] = nhg[0]
            cnhb[0] = nhb[0]
            for i in range(1,256):
                cnhr[i] = nhr[i] + cnhr[i-1]
                cnhg[i] = nhg[i] + cnhg[i-1]
                cnhb[i] = nhb[i] + cnhb[i-1]
            return cnhr,cnhg,cnhb

        elif self.mode == 'L':
            nhgray = self.normilized_histograme()
            cnhgray = np.full(256,0.)
            cnhgray[0] = nhgray[0]
            for i in range(1,256):
                cnhgray[i] = nhgray[i] + cnhgray[i-1]
            return cnhgray
    
    # statistical constants
    def mean(self) -> tuple[float,float,float]|float:
        if self.mode == "RGB":
            return self.r.flatten().mean(),self.g.flatten().mean(),self.b.flatten().mean()
        elif self.mode == "L":
            return self.r.flatten().mean()

    def std(self) -> tuple[float,float,float]|float:
        if self.mode == "RGB":
            return self.r.flatten().std(),self.g.flatten().std(),self.b.flatten().std()
        elif self.mode == "L":
            return self.r.flatten().std()
    
    def median(self) -> tuple[float,float,float]|int:
        if self.mode == "RGB":
            return np.median(self.r.flatten()),np.median(self.g.flatten()),np.median(self.b.flatten())
        elif self.mode == "L":
            return np.median(self.r.flatten())
    
    
    # static functions
    @staticmethod
    def new(w:int,h:int,mode:str):
        """
        create a new image having width w and hight h , and initilise the rgb matrices to zero 
        """
        mode = mode.upper()
        if mode not in MyImage.MODES:
            raise ValueError(f'the selected mode <{mode}> is not provided')   
        v = np.full((h,w),0,dtype=np.uint8)
        return MyImage(v,v,v,mode)
    @staticmethod
    def new_from_pixels(pixels:list[tuple],mode:str,width:int,height:int):
        if len(pixels) == 0:
            raise ValueError("no pixels were given")
        img = MyImage.new(width,height,mode)
        for item in pixels:
            x,y,*v = item
            img[x,y] = v
        return img
    @staticmethod
    def open_image_as_rgb_matrices(path:str):
        """
            Read an image from a file described by the provided path and returns a tuple\n
            which contains 3 matrices representing the values of the image respectevly  R,G,B 
        """
        img = Image.open(path).convert("RGB")
        R,G,B = [],[],[]
        
        for r,g,b in  img.getdata():
            R.append(r),G.append(g),B.append(b)
        
        R = np.array(R).reshape((img.height,img.width))
        G = np.array(G).reshape((img.height,img.width))
        B = np.array(B).reshape((img.height,img.width))

        return MyImage(R,G,B,'RGB')
    
    def save_image(self,path:str):
        self:MyImage = self
        img_to_save = Image.new(self.mode,self.dimensions)
        
        if self.mode == 'L':
            for x in range(self.width):
                for y in range(self.height):
                    v = int(sum(self[x,y])//3)
                    img_to_save.putpixel((x,y),v)
        elif self.mode == 'RGB':
            for x in range(self.width):
                for y in range(self.height):
                    img_to_save.putpixel((x,y),self[x,y])

        
        img_to_save.save(path)
    
    def show_image(self):
        img_to_show = Image.new("RGB",self.dimensions)
        
        for x in range(self.width):
            for y in range(self.height):
                img_to_show.putpixel((x,y),self[x,y])
        plt.imshow(img_to_show)
        plt.show()

    # TODO this function will be re-created cause it is not defined correctly
    def show_histogram(self):
        img = Image.new('RGB',(self.width,self.height))
        img.putdata(list(zip(self.r.flatten(),self.g.flatten(),self.b.flatten())))
        

        if self.mode.upper() == 'RGB':
            axeimg = plt.subplot2grid((2,3),(0,0),colspan=3)
            axeimg.imshow(img)


            axer = plt.subplot2grid((2,3),(1,0),colspan=1)
            axer.hist(self.r.flatten(),bins=256,color='red')

            axeg = plt.subplot2grid((2,3),(1,1),colspan=1)
            axeg.hist(self.g.flatten(),bins=256,color='blue')

            axeb = plt.subplot2grid((2,3),(1,2),colspan=1)
            axeb.hist(self.b.flatten(),bins=256,color='green')

            for i,c in zip([axer,axeg,axeb],['RED','GREEN','BLUE']):
                i.set_title(c)
        
        elif self.mode.upper() == 'L':
            axeimg = plt.subplot2grid((2,1),(0,0))
            axeimg.imshow(img)

            axeg = plt.subplot2grid((2,1),(1,0),colspan=1)
            axeg.hist(np.concatenate([self.r.flatten(),self.b.flatten(),self.g.flatten()]),bins=256,color='gray')
        
        plt.show()
    
    # TODO this function will be re-created cause it is not defined correctly
    def show_normalized_histogram(self):
        img = Image.new('RGB',(self.width,self.height))
        img.putdata(list(zip(self.r.flatten(),self.g.flatten(),self.b.flatten())))

        if self.mode.upper() == 'L':
            G = (self.r.flatten() * MyImage.DEFAUL_GRAY_SCALE_COEF[0] + self.g.flatten() * MyImage.DEFAUL_GRAY_SCALE_COEF[1]\
            + self.b.flatten() * MyImage.DEFAUL_GRAY_SCALE_COEF[2]) / sum(MyImage.DEFAUL_GRAY_SCALE_COEF)  
            values,repetition = np.unique(G,return_counts=True)
            normilized_repetition = repetition/(self.width * self.height)
            
            axe = plt.subplot2grid((2,2),(0,0),colspan=2)
            axe.imshow(img)
            axe.set_title("Source Image")
            
            axe = plt.subplot2grid((2,2),(1,0))
            axe.plot(values,normilized_repetition,color='gray')
            axe.set_title('Normilized Values')

            cummulative = []
            s = 0
            for v in normilized_repetition:
                cummulative.append(s)
                s += v
            
            axe = plt.subplot2grid((2,2),(1,1))
            axe.plot(values,cummulative,color='gray')
            axe.set_title('Cummulated Normilized Values')

        else:
            # ploting the source image
            axe = plt.subplot2grid((4,2),(0,0),colspan=2)
            axe.imshow(img)
            axe.set_title('Source Image')
            # preparing the values
            R = self.r
            values,repetition = np.unique(R,return_counts=True)
            normalized_repetition = repetition/(self.width * self.height)
            
            axe  = plt.subplot2grid((4,2),(1,0))
            axe.plot(values,normalized_repetition,color='red')
            axe.set_title("Normalized Values")

            cummulative = []
            s = 0
            for v in normalized_repetition:
                cummulative.append(s)
                s += v

            axe  = plt.subplot2grid((4,2),(1,1))
            axe.plot(values,cummulative,color='red')
            axe.set_title("Cumulated Normalized Values")

            #----------------------------------------------------------------
            G = self.g
            values,repetition = np.unique(G,return_counts=True)
            normalized_repetition = repetition/(self.width * self.height)
            
            axe  = plt.subplot2grid((4,2),(2,0))
            axe.plot(values,normalized_repetition,color='green')

            cummulative = []
            s = 0
            for v in normalized_repetition:
                cummulative.append(s)
                s += v

            axe  = plt.subplot2grid((4,2),(2,1))
            axe.plot(values,cummulative,color='green')
            #----------------------------------------------------------------
            B = self.b
            values,repetition = np.unique(B,return_counts=True)
            normalized_repetition = repetition/(self.width * self.height)
            
            axe  = plt.subplot2grid((4,2),(3,0))
            axe.plot(values,normalized_repetition,color='blue')

            cummulative = []
            s = 0
            for v in normalized_repetition:
                cummulative.append(s)
                s += v

            axe  = plt.subplot2grid((4,2),(3,1))
            axe.plot(values,cummulative,color='blue')

        plt.show()

    def show_images(images:list):
        DEFAULT_IMAGES_PER_ROW = 3
        number_of_rows = len(images)//DEFAULT_IMAGES_PER_ROW + 1 if len(images) % DEFAULT_IMAGES_PER_ROW != 0 else len(images)
        
        for i,img in enumerate(images):
            axe = plt.subplot2grid((1,len(images)),(0,i)) 
            pil_img = Image.new("RGB",img.dimensions)
            pil_img.putdata(list(zip(img.r.flatten(),img.g.flatten(),img.b.flatten())))
            axe.imshow(pil_img)

        plt.show()
