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
    
    def paste(self,new_width:int,new_height:int,p:tuple[int,int]):
        """
        This function allow you to paste another function on the new image, the new image must be of a larger size
        for it to hold the old image 
        """
        if self.width > new_width or self.height > new_height:
            raise ValueError('The image is bigger than the canvas')
        img = MyImage.new(new_width,new_height,self.mode)

        for x,y,*v in self.pixels():
            if x+p[0] < img.width and y + p[1] < img.height:
                img[x+p[0],y+p[1]] = v if self.mode == 'RGB' else v[0]
            
        return img    
    
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
        # continfication
        raise NotImplementedError()

    # histogram based operations
    def histo_translation(self,i:int):
        nr = (np.array(self.r,dtype=np.int32) + i) % 256
        ng = (np.array(self.g,dtype=np.int32) + i) % 256
        nb = (np.array(self.b,dtype=np.int32) + i) % 256
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
        R = self.r * coef[0]
        G = self.g * coef[1]
        B = self.b * coef[2]
        Gray = np.array((R + G + B) / sum(coef),dtype=np.int8)
        return MyImage(Gray,Gray,Gray,'L')

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

        r_pad = np.pad(self.r,pad_width=size//2,mode='reflect')
        g_pad = np.pad(self.g,pad_width=size//2,mode='reflect')
        b_pad = np.pad(self.b,pad_width=size//2,mode='reflect')

        for x in range(size//2,self.width):
            for y in range(size//2,self.height):
                r = np.array(r_pad[y-size//2:y+size//2+1 , x-size//2:x+size//2+1],dtype=np.int32)
                g = np.array(g_pad[y-size//2:y+size//2+1 , x-size//2:x+size//2+1],dtype=np.int32)
                b = np.array(b_pad[y-size//2:y+size//2+1 , x-size//2:x+size//2+1],dtype=np.int32)
                copy_img.r[y,x] = ((conv_matrix * r).sum())
                copy_img.g[y,x] = ((conv_matrix * g).sum())
                copy_img.b[y,x] = ((conv_matrix * b).sum())
                
        return copy_img

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
    
    # segmentation algorithms
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

            r = []
            for v in clusters.values():
                rr = []
                for p in v:
                    rr.append((p[0],p[1],*self[p[0],p[1]]))
                r.append(rr)
            return r

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

            r = []
            for v in clusters.values():
                rr = []
                for p in v:
                    rr.append((p[0],p[1],*self[p[0],p[1]]))
                r.append(rr)
            return r


    def objects_segmentation(self) -> list:
        ...

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
        
    def remove_outliers(self,metric:str,threash_hold:int|float):
        """
        This function compute the mean and median and std then test if each pixel is between mean - threshold * std and mean + threashold * std if is not then it will replace it buy either the mean of the median  
        """
        metric = metric.upper()
        if threash_hold <= 0:
            raise ValueError("threshold must be an positive number")
        
        if metric not in ('MEAN','MEDIAN'):
            raise ValueError(f"The provided value for metric {metric} is not correct , choose from {('MEAN','MEDIAN')}")
        
        if self.mode == "L":
            mean = self.mean()
            median = self.median()
            std = self.std()
            upper_bound = mean + threash_hold * std
            lower_bound = mean - threash_hold * std
            nv = np.full(self.r.size,0,dtype=np.uint8)
            replcament = mean if metric == "MEAN" else median
            for i,v in enumerate(self.r.flatten()):
                if lower_bound<v<upper_bound:
                    nv[i] = v
                else:
                    nv[i] = replcament
            nv = nv.reshape(self.r.shape)
            return MyImage(nv,nv,nv,"L")
        
        elif self.mode =='RGB':
            mean_r,mean_g,mean_b = self.mean()
            median_r,median_g,median_b = self.median()
            std_r,std_g,std_b = self.std()
            replcament_r,replcament_g,replcament_b = (mean_r,mean_g,mean_b) if metric =="MEAN" else (median_r,median_g,median_b)
            
            lower_bound_r,lower_bound_g,lower_bound_b = mean_r - threash_hold * std_r,mean_g - threash_hold * std_g,mean_b - threash_hold * std_b
            upper_bound_r,upper_bound_g,upper_bound_b = mean_r +  threash_hold* std_r,mean_g + threash_hold * std_g,mean_b +  threash_hold * std_b 

            nv_r = np.full(self.r.size,0,dtype=np.uint8)
            nv_g = np.full(self.r.size,0,dtype=np.uint8)
            nv_b = np.full(self.r.size,0,dtype=np.uint8)
            
            for i,r,g,b in zip(range(self.width*self.height),self.r.flatten(),self.g.flatten(),self.b.flatten()):
                if lower_bound_r < r < upper_bound_r:
                    nv_r[i] = r
                    nv_r[i] = replcament_r

                if lower_bound_g < g < upper_bound_g:
                    nv_g[i] = g
                    nv_g[i] = replcament_g
                
                if lower_bound_b < b < upper_bound_b:
                    nv_b[i] = b
                else:
                    nv_b[i] = replcament_b
            
            return MyImage(nv_r.reshape(self.r.shape),nv_g.reshape(self.g.shape),nv_b.reshape(self.b.shape),self.mode)

        else:
            raise Exception(f"mode {self.mode} is not supported")

    # static functions
    @staticmethod
    def new(w:int,h:int,mode:str):
        """
        create a new image having width w and hight h , and initilise the rgb matrices to zero 
        """
        mode = mode.upper()
        if mode not in MyImage.MODES:
            raise ValueError(f'the mode {mode} is not provided')   
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
