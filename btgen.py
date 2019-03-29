import glob
import cv2
import numpy as np


class BatchGenerator:
    def __init__(self, img_size, imgdir, dirID, num_domains, aug=True):
        self.folderPath = imgdir
        self.imagePath = glob.glob(self.folderPath+"/*")
        self.dirID = dirID
        self.aug = aug
        #self.orgSize = (218,173)
        self.imgSize = (img_size,img_size)

        self.direction = np.arange(num_domains)
        self.direction = np.delete(self.direction,dirID)
        assert self.imgSize[0]==self.imgSize[1]

    def augment(self, img1):
        #軸反転
        if np.random.random() >0.5:
            img1 = cv2.flip(img1,1)

        #軸移動
        rand = (np.random.random()-0.5)/20
        y,x = img1.shape[:2]
        x_rate = x*(np.random.random()-0.5)/20
        y_rate = y*(np.random.random()-0.5)/20
        M = np.float32([[1,0,x_rate],[0,1,y_rate]])
        img1 = cv2.warpAffine(img1,M,(x,y),127)

        #回転
        rand = (np.random.random()-0.5)*5
        M = cv2.getRotationMatrix2D((x/2,y/2),rand,1)

        img1 = cv2.warpAffine(img1,M,(x,y))

        return img1

    def getBatch(self,nBatch,imID,):
        x   = np.zeros( (nBatch,self.imgSize[0],self.imgSize[1],3), dtype=np.float32)
        d = np.random.choice(self.direction,nBatch)
        s = np.ones([nBatch],dtype=np.int32) * self.dirID

        for i,j in enumerate(imID):

            input = cv2.imread(self.imagePath[j])
            input = cv2.resize(input,self.imgSize)
            if self.aug:
                input = self.augment(input)
            x[i,:,:,:] = (input - 127.5) / 127.5

        return x, d ,s
