# -*- coding: utf-8 -*-
"""
    Module for images preprocessing
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from hashlib import sha1
import cv2
from skimage.io import imsave
from skimage import util
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")
import dirs


class ClassifyPreprocess:
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.labels_path = os.path.join(data_path, 'Annotations')
        self.images_path = os.path.join(data_path, r'images')
        self.masks_path = dirs.make_dir(relative_path='masks', top_dir=self.data_path)
        self.names = [name[:-4] for name in os.listdir(self.images_path)]
        
    def make_masks(self):
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
            out_mask = np.zeros(shape=(shape['height'], shape['width'])).astype(np.uint8)
            
            for k, v in bboxes.items():
                lt = (v[0], v[1])
                rb = (v[2], v[3])
                out_mask[lt[1]:rb[1], lt[0]:rb[0]] = 255
            imsave(fname=os.path.join(self.masks_path, f'{name}.png'), arr=out_mask)
            
    def make_classify_dataset(self, window_size=512, step=512, save_images=True, out_dir=None):
        """ Function saves dataset for classifier
        
        :param window_size: Size of the small image window
        :param step: Step for window. If step == window_size, there is no overlap
        :param save_images: Flag to save images
        :param out_dir: Output path to save
        :return:
        """

        non_usual = 0
        for name in tqdm(self.names, total=len(self.names)):
            image = cv2.imread(os.path.join(self.images_path, f'{name}.jpg'))
            mask = cv2.imread(os.path.join(self.masks_path, f'{name}.png'), 0)
            if image.shape[0] != 3000 or image.shape[1] != 4000:
                # print(name)
                non_usual += 1
                continue
            
            # self._draw_images([image, mask])
            h, w = image.shape[0], image.shape[1]
            n_h_win = int(np.round(h / step))
            n_w_win = int(np.round(w / step))

            new_h = step * (n_h_win - 1) + window_size
            new_w = step * (n_w_win - 1) + window_size
            
            top_add = int(np.floor((new_h - h) / 2))
            bot_add = int(np.ceil((new_h - h) / 2))
            left_add = int(np.floor((new_w - w) / 2))
            right_add = int(np.ceil((new_w - w) / 2))
            pad_width = ((top_add, bot_add), (left_add, right_add))
            
            res_image = np.ndarray(shape=(new_h, new_w, 3), dtype=np.uint8)
            for ch in range(res_image.shape[2]):
                img_to_pad = image[:, :, ch]
                res_image[:, :, ch] = util.pad(img_to_pad, pad_width, mode='reflect')
                
            res_mask = util.pad(mask, pad_width, mode='reflect')
            images_zeroes = []
            masks_zeroes = []
            images_obj = []
            masks_obj = []
            for i in range(n_h_win):
                for j in range(n_w_win):
                    h_1, h_2 = i * step, i * step + window_size
                    w_1, w_2 = j * step, j * step + window_size
                    small_image = res_image[h_1:h_2, w_1:w_2, :]
                    small_mask = res_mask[h_1:h_2, w_1:w_2]
                    
                    if np.sum(small_mask) == 0:
                        images_zeroes.append(small_image)
                        masks_zeroes.append(small_mask)
                    else:
                        images_obj.append(small_image)
                        masks_obj.append(small_mask)

            images_zeroes = np.asarray(images_zeroes)
            masks_zeroes = np.asarray(masks_zeroes)
            images_obj = np.asarray(images_obj)
            masks_obj = np.asarray(masks_obj)
            # if images_obj.shape[0] != 0:
            #     self._draw_images([images_obj[0], masks_obj[0]])
            
            n = images_obj.shape[0]
            random_idx = np.random.randint(0, images_zeroes.shape[0], n) if n > 0 else \
                np.random.randint(0, images_zeroes.shape[0], 1)
            
            if save_images:
                for i in range(len(random_idx)):
                    idx = random_idx[i]
                    empty_image = images_zeroes[idx, ...]
                    empty_mask = masks_zeroes[idx, ...]
                    empty_image_name = str(sha1(empty_image).hexdigest()) + '.png'
                    empty_image_path = os.path.join(out_dir, f"images/{empty_image_name}")
                    empty_mask_path = os.path.join(out_dir, f"masks/{empty_image_name}")
                    imsave(empty_image_path, empty_image)
                    imsave(empty_mask_path, empty_mask)
                    
                    if images_obj.shape[0] != 0:
                        # print(images_obj.shape)
                        full_image = images_obj[i, ...]
                        full_mask = masks_obj[i, ...]
                        full_image_name = str(sha1(full_image).hexdigest()) + '.png'
                        full_image_path = os.path.join(out_dir, f"images/{full_image_name}")
                        full_mask_path = os.path.join(out_dir, f"masks/{full_image_name}")
                        imsave(full_image_path, full_image)
                        imsave(full_mask_path, full_mask)

        print(non_usual)
        
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
    # preproc_class.make_masks()
    out_dir = r'/home/ilyado/Programming/proj_resquler_la/data/classify'
    preproc_class.make_classify_dataset(save_images=True, out_dir=out_dir)
