import numpy as np
from PIL import Image
from typing import Self
from matplotlib import pyplot as plt

class MyImage:
    MODES = ('RGB','L')
    FILTERS = ('MEAN')
    DEFAUL_GRAY_SCALE_COEF = (0.299,0.587,0.114)

    def __init__(self,r:np.ndarray,g:np.ndarray,b:np.ndarray,mode) -> None:
        mode:str= mode.upper()
        if mode not in MyImage.MODES:
            raise ValueError(f'Unsupported mode value {mode},mode must be L or RGB')
        self.mode = mode
        if r.ndim != 2 or g.ndim != 2 or b.ndim != 2:
            raise Exception('R,G,B chanels must be in a matrix form')
        if not (r.shape == g.shape and r.shape == b.shape):
            raise Exception('The provided arrays are not cohierant')
        self.r,self.g,self.b = r.copy().astype(np.uint8),g.copy().astype(np.uint8),b.copy().astype(np.uint8)

    @property
    def width(self):
        return len(self.r[0])
    
    @property
    def height(self):
        return len(self.r)

    @property
    def dimensions(self):
        return self.width,self.height

    def pixels(self):
        pixels = []
        if self.mode.upper() == 'RGB':
            for x in range(self.width):
                for y in range(self.height):
                    pixels.append((y,x,self.r[y,x],self.g[y,x],self.b[y,x]))
        elif self.mode.upper() == 'L':
            gray_coef = MyImage.DEFAUL_GRAY_SCALE_COEF
            for x in range(self.width):
                for y in range(self.height):
                    g = int((gray_coef[0] * self.r[y,x] + gray_coef[1]*self.g[y,x] + gray_coef[2] * self.b[y,x])/sum(gray_coef))
                    pixels.append((x,y,g))
        return pixels

    def __getitem__(self,indecies:(int,int)):
        x,y = self.__prepare_indecies__(int(indecies[0]),int(indecies[1]))
        self.__test_indecies__(x,y)
        return self.r[y,x],self.g[y,x],self.b[y,x]
    
    def __setitem__(self,indecies:(int,int),value:tuple[int,int,int]):
        r,g,b = value
        x,y = self.__prepare_indecies__(int(indecies[0]),int(indecies[1]))
        self.__test_indecies__(x,y)
        for v in r,g,b:
            if not 0<=v<256:
                raise Exception('RGB values must be between 0 and 255 inclusive')
        self.r[y,x],self.g[y,x],self.b[y,x] = r,g,b
    
    def __test_indecies__(self,x,y):
        if not 0<=x<self.width:
            raise Exception(f"x value {x}, is greater than the width of the image {self.width}")
        if not 0<=y<self.height:
            raise Exception(f"y value {y}, is greater than the height of the image {self.height}")
    
    def __prepare_indecies__(self,x,y):
        return int(x),int(y)

    def cut(self,x:int,y:int,w:int,h:int):
        x,y = int(x),int(y)
        if not 0<=x<self.width or not 0<=y<self.height:
            raise Exception('The x and y positions are not valide')
        
        w,h = int(w),int(h)
        if not 1<=w<=self.width - x or not 1<=h<self.height - y:
            raise Exception('The new width and the new height must be sub-interval from the image width and height')

        tmp = np.zeros(w * h).reshape((h,w))
        rimg = MyImage(tmp,tmp,tmp,self.mode)
        for _x in range(w):
            for _y in range(h):
                rimg[_x,_y] = self[x+_x,y+_y]
        return rimg 

    def rotate(self,reverse=False):
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
    
    def rotate_simd(self,reverse=False):
        
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

    # filters
    def gray_scale(self):
        coef = MyImage.DEFAUL_GRAY_SCALE_COEF
        R = self.r * coef[0]
        G = self.g * coef[1]
        B = self.b * coef[2]
        Gray = np.array((R + G + B) / sum(coef),dtype=np.int8)
        return MyImage(Gray,Gray,Gray,'L')

    def mean_filter_region(self,size:int):
        if isinstance(size,int):
            if size < 2:
                raise ValueError(f'size must be > 1')
            if size > self.width or size >self.height:
                raise ValueError(f'the provided size is so large')
        else:
            raise ValueError(f"{type(size)} can't be used as a filter")
        
        copy_img = MyImage(np.zeros(self.r.shape),np.zeros(self.r.shape),np.zeros(self.r.shape),self.mode)
        conv_matrix = np.full((size,size),1/(size**2))

        for x in range(size//2,self.width-size ,size):
            for y in range(size//2,self.height-size,size):
                copy_img.r[y:y+size,x:x+size] = np.full((size,size),np.sum(conv_matrix * self.r[y:y+size,x:x+size]))
                copy_img.g[y:y+size,x:x+size] = np.full((size,size),np.sum(conv_matrix * self.g[y:y+size,x:x+size]))
                copy_img.b[y:y+size,x:x+size] = np.full((size,size),np.sum(conv_matrix * self.b[y:y+size,x:x+size]))
        return copy_img


    def mean_filter_center(self,size:int):
        if isinstance(size,int):
            if size < 2:
                raise ValueError(f'size must be > 1')
            if size %2 == 0:
                raise ValueError(f"The size mnust be even")
            if size > self.width or size >self.height:
                raise ValueError(f'the provided size is so large')
        else:
            raise ValueError(f"{type(size)} can't be used as a filter")
        
        copy_img = MyImage(np.zeros(self.r.shape),np.zeros(self.r.shape),np.zeros(self.r.shape),self.mode)
        conv_matrix = np.full((size,size),1/(size**2))
        
        for x in range(size//2,self.width-size//2):
            for y in range(size//2,self.height-size//2):
                copy_img.r[y,x] = (conv_matrix * self.r[y-size//2:y+size//2+1 , x-size//2:x+size//2+1])[size//2,size//2]
                copy_img.g[y,x] = (conv_matrix * self.g[y-size//2:y+size//2+1 , x-size//2:x+size//2+1])[size//2,size//2]
                copy_img.b[y,x] = (conv_matrix * self.b[y-size//2:y+size//2+1, x-size//2:x+size//2+1])[size//2,size//2]
                
        return copy_img


    def gaussian_filter(self,size:int,std:float):
        if isinstance(size,int):
            if size < 2:
                raise ValueError(f'size must be > 1')
            if size > self.width or size >self.height:
                raise ValueError(f'the provided size is so large')
        else:
            raise ValueError(f"{type(size)} can't be used as a filter")
        def gaussian(i,j,size,std):
            tmp = -((i - size // 2)**2 + (j - size //2)**2)/(2*std**2)
            tmp = np.power(np.e,tmp)
            return tmp/(2*np.pi*std**2)
        
        conv_matrix = np.array([[gaussian(i,j,size,std) for i in range(size)] for j in range(size)]) 
        conv_matrix = conv_matrix / np.sum(conv_matrix)    
        copy_img = MyImage(np.zeros(self.r.shape),np.zeros(self.r.shape),np.zeros(self.r.shape),self.mode)

        for x in range(size//2,self.width-size,size):
            for y in range(size//2,self.height-size,size):
                copy_img.r[y:y+size,x:x+size] = conv_matrix * self.r[y:y+size,x:x+size]
                copy_img.g[y:y+size,x:x+size] = conv_matrix * self.g[y:y+size,x:x+size]
                copy_img.b[y:y+size,x:x+size] = conv_matrix * self.b[y:y+size,x:x+size]
        return copy_img

    @staticmethod
    def new(w:int,h:int,mode,defalut_value:tuple[int,int,int]|int) -> Self:
        if mode.upper() == 'RGB':
            r = np.full((h,w),defalut_value[0],dtype=np.uint8)
            g = np.full((h,w),defalut_value[1],dtype=np.uint8)
            b = np.full((h,w),defalut_value[2],dtype=np.uint8)
            return MyImage(r,g,b,mode=mode)
        elif mode.upper() == 'L':
            g = np.full((w,h),defalut_value,dtype=np.uint8)
            return MyImage(g,g,g,mode=mode)
        else:
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
    
    @staticmethod
    def show_image(img):
        img:MyImage = img
        img_to_show = Image.new("RGB",img.dimensions)
        
        for x in range(img.width):
            for y in range(img.height):
                img_to_show.putpixel((x,y),img[x,y])
        plt.imshow(img_to_show)
        plt.show()

    def histogram(self):
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
    
    def normalized_histogram(self):
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

    def heat_map(self):
        if self.mode.upper() == 'L':
            G = (self.r.flatten() * MyImage.DEFAUL_GRAY_SCALE_COEF[0] + self.g.flatten() * MyImage.DEFAUL_GRAY_SCALE_COEF[1]\
            + self.b.flatten() * MyImage.DEFAUL_GRAY_SCALE_COEF[2]) / sum(MyImage.DEFAUL_GRAY_SCALE_COEF)