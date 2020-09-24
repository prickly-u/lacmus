import cv2
from pathlib import Path
import xml.etree.ElementTree as ET
import os
import random

def create_crop_annotation(
        crop_name, crop_height, crop_width, bbox_ymin, bbox_xmin, bbox_ymax, bbox_xmax):
    xsi = "http://www.w3.org/2001/XMLSchema-instance"
    xsd = "http://www.w3.org/2001/XMLSchema"
    ns = {"xmlns:xsi": xsi, "xmlns:xsd": xsd}
    annotation = ET.Element('annotation', ns)

    folder = ET.SubElement(annotation, 'folder')
    folder.text = 'VocGalsTfl'
    filename = ET.SubElement(annotation, 'filename')
    filename.text = crop_name

    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'

    size = ET.SubElement(annotation, 'size')
    height = ET.SubElement(size, 'height')
    height.text = str(crop_height)
    width = ET.SubElement(size, 'width')
    width.text = str(crop_width)
    depth = ET.SubElement(size, 'depth')
    depth.text = str(3)

    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = str(0)

    object = ET.SubElement(annotation, 'object')
    name = ET.SubElement(object, 'name')
    name.text = 'Pedestrian'
    pose = ET.SubElement(object, 'pose')
    pose.text = 'Unspecified'
    truncated = ET.SubElement(object, 'truncated')
    truncated.text = str(0)
    difficult = ET.SubElement(object, 'difficult')
    difficult.text = str(0)

    bndbox = ET.SubElement(object, 'bndbox')
    ymin = ET.SubElement(bndbox, 'ymin')
    ymin.text = str(bbox_ymin)
    xmin = ET.SubElement(bndbox, 'xmin')
    xmin.text = str(bbox_xmin)
    ymax = ET.SubElement(bndbox, 'ymax')
    ymax.text = str(bbox_ymax)
    xmax = ET.SubElement(bndbox, 'xmax')
    xmax.text = str(bbox_xmax)

    return annotation

def create_convert_imagesets(original_folder, crops_folder):
    imagesets_subfolder = 'ImageSets/Main'
    images_subfolder = 'JPEGImages'
    cropped_imagesets_folder = Path(crops_folder, imagesets_subfolder)
    if not os.path.exists(cropped_imagesets_folder):
        os.mkdir(cropped_imagesets_folder)
        print('Created folder: ', cropped_imagesets_folder)

    crops = os.listdir(Path(crops_folder, images_subfolder))
    imagesets = os.listdir(Path(original_folder, imagesets_subfolder))
    for imageset in imagesets:
        with open(Path(original_folder, imagesets_subfolder, imageset)) as infile:
            image_numbers = [line.strip('\n') for line in infile.readlines()]
            with open(Path(crops_folder, imagesets_subfolder, imageset), 'w') as outfile:
                for image_number in image_numbers:
                    image_crops = [crop[:-4] + '\n' for crop in crops if crop.startswith(image_number + '_')]
                    outfile.writelines(image_crops)


def main():
    
    # open CFG file, define paths and crops shapes
    config = open('config.cfg')
    for line in config:
        if line.startswith('CROP_SIZE'):
            crop_size = int(line.split('=')[1].strip().replace('\n',''))
        if line.startswith('ORIGINAL_DATASET_PATH'):
            original_folder = Path(line.split('=')[1].strip().replace('\n',''))
        if line.startswith('CROPPED_DATASET_PATH'):
            crops_folder = line.split('=')[1].strip().replace('\n','')
            
    images_folder      =         'JPEGImages'
    annotations_folder =         'Annotations'
    config.close()

    '''print('Dataset location: ', original_folder)
    
    # Create folders for outputs
    if not os.path.exists(crops_folder):
        os.mkdir(crops_folder)
        print('Created folder: ', crops_folder)

    print('Processing started...')

    # parse each annotation file
    annotations_list = os.listdir(Path(original_folder, annotations_folder))
    n_files = len(annotations_list)
    passed_files=1

    for filename in annotations_list:
        
        if not filename.endswith('.xml'):
            continue
            
        fullname = Path(original_folder, annotations_folder, filename)
        tree = ET.parse(fullname)    
        root = tree.getroot()    
        bbox_num = 0

        img = cv2.imread(str(Path(original_folder, images_folder, filename[:-3]+'jpg')))
    
        for rec in root:
            
            # get source image size
            if rec.tag == 'size': 
                height = int(rec.findtext('height'))
                width = int(rec.findtext('width'))
                
            # list all available bboxes        
            if rec.tag == 'object': 
                for box in rec:
                    if box.tag=='bndbox':
                        
                        # get initial bbox corners
                        ymin = int(box.findtext('ymin'))
                        ymax = int(box.findtext('ymax'))
                        xmin = int(box.findtext('xmin'))
                        xmax = int(box.findtext('xmax'))
                        
                        # calculate necessary padding to get crop of crop_size
                        padding_w = int((crop_size - (xmax - xmin))/2.)
                        padding_h = int((crop_size - (ymax - ymin))/2.)
                        
                        # get random shift within 25% of crop_size from bbox center
                        random_dx = int((random.random()-.5)*.5*crop_size)
                        random_dy = int((random.random()-.5)*.5*crop_size)
                        
                        # calculate crop corners
                        new_xmin = xmin - padding_w + random_dx
                        new_xmax = xmax + padding_w + random_dx
                        new_ymin = ymin - padding_h + random_dy
                        new_ymax = ymax + padding_h + random_dy
                        
                        # do not proceed if crop is outside of image
                        if (new_xmin<1 or new_xmax>width-1 or new_ymin<1 or new_ymax>height-1):continue
                        
                        dx = new_xmax - new_xmin
                        dy = new_ymax - new_ymin
                        
                        # correct crop corners to get exact crop_size
                        if dx<crop_size:
                            if ((new_xmax+new_xmin)/2.)<(width/2.):
                                new_xmax+=1
                            else:
                                new_xmin-=1
                                
                        if dy<crop_size:
                            if ((new_ymax+new_ymin)/2.)<(height/2.):
                                new_ymax+=1
                            else:
                                new_ymin-=1
                       
                        # create crop
                        crop = img.copy()
                        crop = crop[new_ymin:new_ymax, new_xmin:new_xmax]
                        
                        # save all this stuff
                        crop_name = filename[:-4] + '_' + str(bbox_num)
                        crop_path = str(Path(crops_folder, images_folder, crop_name + '.jpg'))
                        cv2.imwrite(crop_path, crop)

                        new_bbox_ymin = ymin - new_ymin
                        new_bbox_ymax = ymax - new_ymin
                        new_bbox_xmin = xmin - new_xmin
                        new_bbox_xmax = xmax - new_xmin
                        new_height = new_xmax - new_xmin
                        new_width = new_ymax - new_ymin
                        annotation = create_crop_annotation(
                            crop_name,
                            new_height, new_width,
                            new_bbox_ymin, new_bbox_xmin, new_bbox_ymax, new_bbox_xmax)

                        annotation_path = str(Path(crops_folder, annotations_folder, crop_name + '.xml'))
                        ET.ElementTree(annotation).write(annotation_path)
                        
                        # goto next bbox in current file
                        bbox_num = bbox_num + 1
                        
        if passed_files in range(int(n_files/10), n_files, int(n_files/10)):
            print(str(int(passed_files/n_files*10*10)+1)+'% done...')
        passed_files+=1

    print('Croping completed.')
'''
    print('Creating trainsets of cropped images.')
    create_convert_imagesets(original_folder, crops_folder)
    print('Trainsets created.')

                            
if __name__=='__main__':
    main()
