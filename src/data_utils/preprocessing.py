# -*- coding: utf-8 -*-
"""
    Module for images preprocessing
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import cv2
from skimage.io import imsave
from tqdm import tqdm
import matplotlib.pyplot as plt


class ClassifyPreprocess:
    
    def __init__(self, data_path):
        # self.data_path = data_path
        
        self.labels_path = os.path.join(data_path, 'Annotations')
        self.images_path = os.path.join(data_path, r'images')
        self.names = [name[:-4] for name in os.listdir(self.images_path)]
        print(self.names)
        
    def get_masks(self):
        """ Function make binary masks for images
        
        :return:
        """
        for name in tqdm(self.names, total=len(self.names)):
            label = name + '.xml'
            mydoc = ET.parse(os.path.join(self.labels_path, label))
            root = mydoc.getroot()
            shape = dict()
            bboxes = dict()
            obj_cnt = 0
            for elem in root:
                if elem.tag == 'size':
                    for child in elem:
                        shape[child.tag] = int(child.text)
                elif elem.tag == 'object':
                    bbox_list = []
                    for child in elem:
                        if child.tag == 'bndbox':
                            for e in child:
                                bbox_list.append(int(e.text))
                        bboxes[obj_cnt] = bbox_list
                    obj_cnt += 1
            out_mask = np.zeros(shape=(shape['height'], shape['width']))
            
            for k, v in bboxes.items():
                lt = (v[0], v[1])
                rb = (v[2], v[3])
                out_mask[lt[1]:rb[1], lt[0]:rb[0]] = 255
        
    @staticmethod
    def _draw_images(images_list):
        n_images = len(images_list)
        fig = plt.figure(figsize=(15, 10))
        for i, image in enumerate(images_list):
            ax = fig.add_subplot(1, n_images, i + 1)
            ax.imshow(image)
        plt.show()
        
        
        
if __name__ == '__main__':
    data_folder = f'/home/ilyado/Programming/proj_resquler_la/data/LizaAlertDroneDatasetV1'
    preproc_class = ClassifyPreprocess(data_path=data_folder)
    preproc_class.get_masks()
        
        