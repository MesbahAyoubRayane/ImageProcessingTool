import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

class MyImage:
    """
    A class containging all the functions learned in the image processing class:
    support two types of images 'RGB' an 'L' (gray scale)
    """
    MODES = ('RGB','L') 
    DEFAUL_GRAY_SCALE_COEF = (0.299,0.587,0.114) # this are the deafults values for transforming from rgb to gray scale

    def __init__(self,r:np.ndarray,g:np.ndarray,b:np.ndarray,mode:str):
        """
        r,g,b are numpy matrices the must have the same shape (width*height)
        mode : 'RGB' or 'L'
        """
        mode = mode.upper()
        if mode not in MyImage.MODES:
            raise ValueError(f'Unsupported mode value {mode},mode must be L or RGB')
        self.mode = mode
        if r.ndim != 2 or g.ndim != 2 or b.ndim != 2:
            raise Exception('R,G,B chanels must be in a matrix form')
        if not (r.shape == g.shape and r.shape == b.shape):
            raise Exception('The provided arrays are not cohierant')
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
                for y in range(self.height):
                    yield y,x,self.r[y,x],self.g[y,x],self.b[y,x]
        elif self.mode.upper() == 'L':
            gray_coef = MyImage.DEFAUL_GRAY_SCALE_COEF
            for x in range(self.width):
                for y in range(self.height):
                    g = int((gray_coef[0] * self.r[y,x] + gray_coef[1]*self.g[y,x] + gray_coef[2] * self.b[y,x])/sum(gray_coef))
                    yield x,y,g

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
            r,g,b = value    
            for v in r,g,b:
                if not isinstance(v,int):
                    raise ValueError(f"the provided value is not an integer v={v}")
                if not 0<=v<256:
                    raise ValueError('RGB values must be between 0 and 255 inclusive')
            self.r[y,x],self.g[y,x],self.b[y,x] = r,g,b
        
        elif self.mode.upper() == 'L':
            if not isinstance(value,int):
                raise ValueError(f"the provided value is not an integer v={v}")
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
        if w <= 0 or h <= 0:
            raise ValueError(f"the width and height must be a positive value , w = {w},h = {h}")
        
        tmp = np.zeros(w * h).reshape((h,w))
        rimg = MyImage(tmp,tmp,tmp,self.mode)
        for _x in range(w):
            for _y in range(h):
                if 0<=_x<self.width and 0<=_y<self.height:
                    rimg[_x,_y] = self[x+_x,y+_y]
        return rimg 

    def rotate_90_degrees(self,reverse=False):
        """
            This method rotate the image and return a new MyImage,if reverse=False it rotate in the clock wise direction
        """
        tmp = np.zeros(self.width * self.height,dtype=np.uint8).reshape((self.width,self.height))
        rimg = MyImage(tmp,tmp,tmp,mode=self.mode)
        if not reverse:
            for x in range(self.width):
                for y in range(self.height):
                    rimg[self.height - y - 1,self.width - x - 1] = self[x,y]
        else:
            for x in range(self.width):
                for y in range(self.height):
                    rimg[y,self.width - x - 1] = self[x,y]
        return rimg
    
    def rotate_90_degrees_simd(self,reverse=False):
        """simd function are optimized using numpy vectorial operations"""
        if not reverse:
            r = np.array([self.r[:,i][::-1] for i in range(self.width)])
            g = np.array([self.g[:,i][::-1] for i in range(self.width)])
            b = np.array([self.b[:,i][::-1] for i in range(self.width)])
        else:
            r = np.array([self.r[:,i] for i in range(self.width)])
            g = np.array([self.g[:,i] for i in range(self.width)])
            b = np.array([self.b[:,i] for i in range(self.width)])
            
        return MyImage(r,g,b,self.mode)
    
    def flip(self,axe:str):
        axe = axe.lower()
        if axe not in ('h','v'):raise Exception("axe must be v or h")
        tmp =np.zeros(self.width*self.height).reshape(self.r.shape)
        fimg = MyImage(tmp,tmp,tmp,self.mode)
        
        if axe == 'v':
            for x in range(self.width):
                for y in range(self.height):  
                    fimg[self.width - x - 1,y] = self[x,y]
        elif axe == 'h':
            for x in range(self.width):
                for y in range(self.height):  
                    fimg[x, self.height - y - 1] = self[x,y]
        
        return fimg
    
    def flip_simd(self,axe:str):
        """
        flip_simd : as the name suggestes , it allow verticale and horizantal fliping of the image , it  uses the numpy array simd operation to accelerate the process  
        """
        axe = axe.lower()
        if axe not in ('h','v'):raise Exception("axe must be v or h")
        tmp =np.zeros(self.width*self.height).reshape(self.r.shape)
        
        if axe == 'v':
            fimg = MyImage(
                np.array([row[::-1] for row in self.r]),
                np.array([row[::-1] for row in self.g]),
                np.array([row[::-1] for row in self.b]),self.mode)
        elif axe == 'h':
            fimg = MyImage(np.array([row for row in self.r[::-1]]),np.array([row for row in self.g[::-1]]),np.array([row for row in self.b[::-1]]),self.mode)

        return fimg
    
    def rotate(self,theta:float|int):
        """ theta must be in degrees """
        X1 = np.arange(len(self.r[0]))
        Y1 = np.arange(len(self.r))
        X0,Y0 = len(self.r[0])/2,len(self.r)/2
        
        r = np.zeros(self.r.shape)
        g = np.zeros(self.g.shape)
        b = np.zeros(self.b.shape)

        theta = (theta * 2 * np.pi)/360
        SIN_THETA,COS_THETA = np.sin(theta),np.cos(theta)
        for x in X1:
            for y in Y1:
                x2 = int((x - X0)*COS_THETA - (y - Y0)*SIN_THETA + X0)
                y2 = int(SIN_THETA * (x - X0) + COS_THETA*(y - Y0) + Y0)
                if not 0<=x2<len(self.r[0]) or not  0<=y2<len(self.r): continue
                r[y2,x2] = self.r[y,x]
                g[y2,x2] = self.g[y,x]
                b[y2,x2] = self.b[y,x]
        
        return MyImage(r,g,b,self.mode)
    
    # histigram based operations
    def histo_translation(self,i:int):
        nr = (np.array(self.r,dtype=np.int32) + i) % 256
        ng = (np.array(self.g,dtype=np.int32) + i) % 256
        nb = (np.array(self.b,dtype=np.int32) + i) % 256
        return MyImage(nr,ng,nb,self.mode)
    
    def inverse(self):
        r = 255 - np.array(self.r)
        g = 255 - np.array(self.g)
        b = 255 - np.array(self.b)
        return MyImage(r,g,b,self.mode) 
    
    def expansion_dynamique(self):
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

    # filters
    def gray_scale(self):
        coef = MyImage.DEFAUL_GRAY_SCALE_COEF
        R = self.r * coef[0]
        G = self.g * coef[1]
        B = self.b * coef[2]
        Gray = np.array((R + G + B) / sum(coef),dtype=np.int8)
        return MyImage(Gray,Gray,Gray,'L')

    def red_scale(self):
        z = np.zeros(self.r.shape)
        return MyImage(self.r,z,z,"RGB")
    
    def blue_scale(self):
        z = np.zeros(self.r.shape)
        return MyImage(z,z,self.b,"RGB")
    
    def green_scale(self):
        z = np.zeros(self.r.shape)
        return MyImage(z,self.g,z,"RGB")

    def mean_filter(self,size:int):
        if isinstance(size,int):
            if size < 2:
                raise ValueError(f'size must be > 1')
            if size %2 == 0:
                raise ValueError(f"The size must be odd number")
            if size > self.width or size >self.height:
                raise ValueError(f'the provided size is so large')
        else:
            raise ValueError(f"{type(size)} can't be used as a filter")
        
        copy_img = self.copy()
        conv_matrix = np.full((size,size),1/(size**2))
        
        for x in range(size//2,self.width-size//2):
            for y in range(size//2,self.height-size//2):
                r = np.array(self.r[y-size//2:y+size//2+1 , x-size//2:x+size//2+1],dtype=np.int32)
                g = np.array(self.g[y-size//2:y+size//2+1 , x-size//2:x+size//2+1],dtype=np.int32)
                b = np.array(self.b[y-size//2:y+size//2+1 , x-size//2:x+size//2+1],dtype=np.int32)
                copy_img.r[y,x] = ((conv_matrix * r).sum())
                copy_img.g[y,x] = ((conv_matrix * g).sum())
                copy_img.b[y,x] = ((conv_matrix * b).sum())
                
        return copy_img

    # TODO this function doesn't work properly
    def gaussian_filter(self,size:int,std:float,CONST=10):
        if isinstance(size,int):
            if size < 2:
                raise ValueError(f'size must be > 1')
            if size > self.width or size >self.height:
                raise ValueError(f'the provided size is so large')
            if size %2 == 0:
                raise ValueError(f"The size must be odd number")
        else:
            raise ValueError(f"{type(size)} can't be used as a filter")
        def gaussian(i,j,size,std):
            tmp = -((i - size // 2)**2 + (j - size //2)**2)/(2*std**2)
            tmp = np.power(np.e,tmp)
            return CONST*tmp/(2*np.pi*std**2)
        
        conv_matrix = np.array([[gaussian(i,j,size,std) for i in range(size)] for j in range(size)],dtype=np.int8)
        #conv_matrix = conv_matrix / np.sum(conv_matrix)  
        copy_img = self.copy()

        for x in range(size//2,self.width-size//2):
            for y in range(size//2,self.height-size//2):
                try:
                    r = np.array(self.r[y-size//2:y+size//2+1 , x-size//2:x+size//2+1],dtype=np.int32)
                    g = np.array(self.g[y-size//2:y+size//2+1 , x-size//2:x+size//2+1],dtype=np.int32)
                    b = np.array(self.b[y-size//2:y+size//2+1 , x-size//2:x+size//2+1],dtype=np.int32)
                    copy_img.r[y,x] = int((conv_matrix * r)[size//2,size//2]%256)
                    copy_img.g[y,x] = int((conv_matrix * g)[size//2,size//2]%256)
                    copy_img.b[y,x] = int((conv_matrix * b)[size//2,size//2]%256)
                except Exception as e:
                    print(x,y)
        return copy_img

    # Haitem's Codes:
    def gaussian_filter(self, size: int, std: float):
        if isinstance(size, int):
            if size < 2:
                raise ValueError(f'size must be > 1')
            if size > self.width or size > self.height:
                raise ValueError(f'the provided size is too large')
        else:
            raise ValueError(f"{type(size)} can't be used as a filter")

        # Create a Gaussian kernel using NumPy
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        kernel = np.exp(-((x - size // 2) ** 2 + (y - size // 2) ** 2) / (2 * std ** 2))
        kernel /= (2 * np.pi * std ** 2)

        # Normalize the kernel
        kernel /= kernel.sum()

        # Pad the input image using NumPy
        """
        Padding an image is a common practice in image processing when you want to apply convolution or filtering operations 
        Padding involves adding extra pixels around the edges of the image 
        to ensure that the filter kernel can be applied to all the pixels, even those at the image boundary
        """
        extended_r = np.pad(self.r, ((size // 2, size // 2), (size // 2, size // 2)), 'reflect')
        extended_g = np.pad(self.g, ((size // 2, size // 2), (size // 2, size // 2)), 'reflect')
        extended_b = np.pad(self.b, ((size // 2, size // 2), (size // 2, size // 2)), 'reflect')

        copy_img = self.copy()

        for x in range(size // 2, self.width - size // 2):
            for y in range(size // 2, self.height - size // 2):
                r_patch = extended_r[y - size // 2:y + size // 2 + 1, x - size // 2:x + size // 2 + 1]
                g_patch = extended_g[y - size // 2:y + size // 2 + 1, x - size // 2:x + size // 2 + 1]
                b_patch = extended_b[y - size // 2:y + size // 2 + 1, x - size // 2:x + size // 2 + 1]

                # Apply the Gaussian filter using element-wise operations
                r_filtered = np.sum(r_patch * kernel)
                g_filtered = np.sum(g_patch * kernel)
                b_filtered = np.sum(b_patch * kernel)

                copy_img.r[y, x] = int(np.clip(r_filtered, 0, 255))
                copy_img.g[y, x] = int(np.clip(g_filtered, 0, 255))
                copy_img.b[y, x] = int(np.clip(b_filtered, 0, 255))

        return copy_img

    def color_segmt(self, threshold: int):
        kernel1 = np.full(shape=(3, 3), fill_value=0)
        kernel1[:, 0] = -1
        kernel1[:, -1] = 1

        kernel2 = np.full(shape=(3, 3), fill_value=0)
        kernel2[0, :] = -1
        kernel2[-1, :] = 1

        size = 3

        # Pad the input image using NumPy
        extended_r = np.pad(self.r, ((size // 2, size // 2), (size // 2, size // 2)), 'reflect')
        extended_g = np.pad(self.g, ((size // 2, size // 2), (size // 2, size // 2)), 'reflect')
        extended_b = np.pad(self.b, ((size // 2, size // 2), (size // 2, size // 2)), 'reflect')

        # Create a new image to store the processed data
        segmented_img = self.copy()

        for x in range(size // 2, self.width - size // 2):
            for y in range(size // 2, self.height - size // 2):
                r_patch = extended_r[y - size // 2:y + size // 2 + 1, x - size // 2:x + size // 2 + 1]
                g_patch = extended_g[y - size // 2:y + size // 2 + 1, x - size // 2:x + size // 2 + 1]
                b_patch = extended_b[y - size // 2:y + size // 2 + 1, x - size // 2:x + size // 2 + 1]

                # Apply the Gaussian filter using element-wise operations (Convolution)
                r_i = np.sum(r_patch * kernel1)
                g_i = np.sum(g_patch * kernel1)
                b_i = np.sum(b_patch * kernel1)

                r_j = np.sum(r_patch * kernel2)
                g_j = np.sum(g_patch * kernel2)
                b_j = np.sum(b_patch * kernel2)

                # Calculate the magnitude of the gradients
                Gr_new = np.sqrt(np.square(r_i) + np.square(r_j))
                Gg_new = np.sqrt(np.square(g_i) + np.square(g_j))
                Gb_new = np.sqrt(np.square(b_i) + np.square(b_j))

                # Apply thresholding
                if Gr_new > threshold:
                    segmented_img.r[y, x] = 255
                else:
                    segmented_img.r[y, x] = 0

                if Gg_new > threshold:
                    segmented_img.g[y, x] = 255
                else:
                    segmented_img.g[y, x] = 0

                if Gb_new > threshold:
                    segmented_img.b[y, x] = 255
                else:
                    segmented_img.b[y, x] = 0

        return segmented_img
    
    def heat_map(self):
        # from the top
        delta_r_t = np.zeros(self.r.shape,dtype=np.int32)
        delta_g_t = np.zeros(self.g.shape,dtype=np.int32)
        delta_b_t = np.zeros(self.b.shape,dtype=np.int32)
        for i in range(len(self.r)-2):
            delta_r_t[i+1] = self.r[i+2] - self.r[i]
            delta_g_t[i+1] = self.g[i+2] - self.g[i]
            delta_b_t[i+1] = self.b[i+2] - self.b[i]
    
        # from the left
        delta_r_l = np.zeros(self.r.shape,dtype=np.int32)
        delta_g_l = np.zeros(self.g.shape,dtype=np.int32)
        delta_b_l = np.zeros(self.b.shape,dtype=np.int32)
        
        for i in range(0,len(self.r[0]) - 2,2):
            delta_r_l[:,i+1] = self.r[:,i+2] - self.r[:,i]
            delta_g_l[:,i+1] = self.g[:,i+2] - self.r[:,i]
            delta_b_l[:,i+1] = self.b[:,i+2] - self.r[:,i]  

        delta_r = np.zeros(self.r.shape,dtype=np.int32)
        delta_g = np.zeros(self.g.shape,dtype=np.int32)
        delta_b = np.zeros(self.b.shape,dtype=np.int32)
        
        for i in range(len(self.r)):
            for j in range(len(self.r[0])):
                x  = delta_r_l[i,j] if abs(delta_r_l[i,j]) > abs(delta_r_t[i,j]) else delta_r_t[i,j]
                delta_r[i,j] = x
                x  = delta_g_l[i,j] if abs(delta_g_l[i,j]) > abs(delta_g_t[i,j]) else delta_g_t[i,j]
                delta_g[i,j] = x
                x  = delta_b_l[i,j] if abs(delta_b_l[i,j]) > abs(delta_b_t[i,j]) else delta_b_t[i,j]
                delta_b[i,j] = x
        
        delta_r = (delta_r + 255)//2
        delta_g = (delta_g + 255)//2
        delta_b = (delta_b + 255)//2
        return MyImage(delta_r,delta_g,delta_b,self.mode)
    

    def create_histograme(self) -> np.ndarray|tuple[np.ndarray,np.ndarray,np.ndarray]:
        if self.mode == "RGB":
            hr = np.full((256,),fill_value=0,dtype=np.int32)
            hg = np.full((256,),fill_value=0,dtype=np.int32)
            hb = np.full((256,),fill_value=0,dtype=np.int32)
            for ri,gi,bi in zip(self.r.flatten(),self.g.flatten(),self.b.flatten()):
                hr[ri] += 1
                hg[gi] += 1
                hb[bi] += 1
            
            return hr,hg,hb

        elif self.mode == "L":
            hgray = np.full((256,),fill_value=0,dtype=np.int32)
            for v in self.r.flatten():
                hgray[v] += 1
            return hgray
    
    def create_cumulated_histograme(self) ->  np.ndarray|tuple[np.ndarray,np.ndarray,np.ndarray]:
        if self.mode == "RGB":
            hr,hg,hb = self.create_histograme()
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
            hgray = self.create_histograme()
            chgray = np.full((256,),dtype=np.int32,fill_value=0)
            sum_gray = 0
            for i in range(256):
                sum_gray += hgray[i]
                chgray[i] = sum_gray
            return chgray

    
    def create_normilized_histograme(self) -> np.ndarray|tuple[np.ndarray,np.ndarray,np.ndarray]:
        if self.mode == "RGB":
            hr,hg,hb = self.create_histograme()
            nhr,nhg,nhb = np.array(hr,dtype=np.float64),np.array(hg,dtype=np.float64),np.array(hb,dtype=np.float64) 
            nhr /= (self.width*self.height)
            nhg /= (self.width*self.height)
            nhb /= (self.width*self.height)
            return  nhr,nhg,nhb
        elif self.mode == 'L':
            hgray = self.create_histograme()
            nhgray = np.array(hgray,dtype=np.float64) / (self.width*self.height)
            return nhgray
    
    def create_cumulative_normilized_histograme(self) -> np.ndarray|tuple[np.ndarray,np.ndarray,np.ndarray]:
        if self.mode == 'RGB':
            nhr,nhg,nhb = self.create_normilized_histograme()
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
            nhgray = self.create_normilized_histograme()
            cnhgray = np.full(256,0.)
            cnhgray[0] = nhgray[0]
            for i in range(1,256):
                cnhgray[i] = nhgray[i] + cnhgray[i-1]
            return cnhgray

    # static functions
    @staticmethod
    def new(w:int,h:int,mode:str):
        """
        create a new image having width w and hight h , and initilise the rgb matrices to zero 
        """
        if mode.upper() == 'RGB':
            r = np.full((h,w),defalut_value[0],dtype=np.uint8)
            g = np.full((h,w),defalut_value[1],dtype=np.uint8)
            b = np.full((h,w),defalut_value[2],dtype=np.uint8)
            return MyImage(r,g,b,mode=mode)
        elif mode.upper() == 'L':
            g = np.full((w,h),defalut_value,dtype=np.uint8)
            return MyImage(g,g,g,mode=mode)
        mode = mode.upper()
        if mode not in MyImage.MODES:
            raise ValueError(f'the mode {mode} is not provided')

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
    
    @staticmethod
    def save_image(img,path:str):
        img:MyImage = img
        img_to_save = Image.new(img.mode,img.dimensions)
        
        if img.mode == 'L':
            for x in range(img.width):
                for y in range(img.height):
                    v = int(sum(img[x,y])//3)
                    img_to_save.putpixel((x,y),v)
        elif img.mode == 'RGB':
            for x in range(img.width):
                for y in range(img.height):
                    img_to_save.putpixel((x,y),img[x,y])

        
        img_to_save.save(path)
    
    def show_image(self):
        img_to_show = Image.new("RGB",self.dimensions)
        
        for x in range(self.width):
            for y in range(self.height):
                img_to_show.putpixel((x,y),self[x,y])
        plt.imshow(img_to_show)
        plt.show()

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
