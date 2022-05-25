# coding: utf-8
# The code is written by Linghui

import numpy as np
from skimage import data
from matplotlib import pyplot as plt
import get_glcm
import time
from PIL import Image
import os
import glob
import cv2
import pandas as pd

def main():
    pass


if __name__ == '__main__':
    
    main()
     
    start = time.time()

    print('---------------0. Parameter Setting-----------------')
    nbit = 64 # gray levels
    mi, ma = 0, 255 # max gray and min gray
    slide_window = 7 # sliding window
    # step = [2, 4, 8, 16] # step
    # angle = [0, np.pi/4, np.pi/2, np.pi*3/4] # angle or direction
    step = [2]
    angle = [0]
    print('-------------------1. Load Data---------------------')
    List = []
    for i in range(1,3):
        image = rf"G:/Hyperspectral_Image/PaviaU/paviaU/paviaUB{i}.tif"
        img = np.array(Image.open(image)) # If the image has multi-bands, it needs to be converted to grayscale image
        img = np.uint8(255.0 * (img - np.min(img))/(np.max(img) - np.min(img))) # normalization
        h, w = img.shape
        print('------------------2. Calcu GLCM---------------------')
        glcm = get_glcm.calcu_glcm(img, mi, ma, nbit, slide_window, step, angle)
        print('-----------------3. Calcu Feature-------------------')
        # 
        for i in range(glcm.shape[2]):
            for j in range(glcm.shape[3]):
                glcm_cut = np.zeros((nbit, nbit, h, w), dtype=np.float32)
                glcm_cut = glcm[:, :, i, j, :, :]
                mean = get_glcm.calcu_glcm_mean(glcm_cut, nbit)
                # variance = get_glcm.calcu_glcm_variance(glcm_cut, nbit)
                # homogeneity = get_glcm.calcu_glcm_homogeneity(glcm_cut, nbit)
                # contrast = get_glcm.calcu_glcm_contrast(glcm_cut, nbit)
                # dissimilarity = get_glcm.calcu_glcm_dissimilarity(glcm_cut, nbit)
                # entropy = get_glcm.calcu_glcm_entropy(glcm_cut, nbit)
                # energy = get_glcm.calcu_glcm_energy(glcm_cut, nbit)
                # correlation = get_glcm.calcu_glcm_correlation(glcm_cut, nbit)
                # Auto_correlation = get_glcm.calcu_glcm_Auto_correlation(glcm_cut, nbit)
        Mean = mean.reshape(-1,1)
        List.append(Mean)
    
        

        Mean1 = pd.DataFrame(List)
        print('---------------4. save to csv----------------')
        Mean = mean.reshape(-1,103)
    Variance =  variance.reshape(-1,1)



    


    end = time.time()
    print('Code run time:', end - start)
