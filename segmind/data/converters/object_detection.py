#!/usr/bin/python

# import cv2
import glob
import json
import numpy as np
import os
import os.path
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pascal_voc_writer import Writer
from pycocotools.coco import COCO

from segmind.data.converters.pascal_voc_io import (PascalVocReader,
                                                   PascalVocWriter)
from segmind.data.converters.utils import get, get_and_check, process_file
from segmind.data.converters.yolo_io import YoloReader, YOLOWriter

# TODO: @shreeram
# use glob.glob in place of os.listdir
# use os.path.join() in place of + "/" +

START_BOUNDING_BOX_ID = 1
PRE_DEFINED_CATEGORIES = {}


def coco_to_voc_bbox(bbox):

    assert isinstance(bbox, np.ndarray) and bbox.ndim == 2 and bbox.shape[
        1] == 4, f'bbox should be numpy of dimension (Nx4), got {bbox.shape}'

    bbox[:, 2] = bbox[:, 2] + bbox[:, 0]
    bbox[:, 3] = bbox[:, 3] + bbox[:, 1]

    return bbox


# def yolo_to_voc_bbox(image, bbox):
#
#     image_array = cv2.imread(image)
#     assert isinstance(bbox, np.ndarray) and bbox.ndim == 2 and bbox.shape[
#         1] == 4, 'bbox should be numpy of dimension (Nx4)'
#     height, width, c = image_array.shape
#
#     bbox_widths = bbox[:, 2] * width
#     bbox_heights = bbox[:, 3] * height
#
#     bbox[:, 0] = bbox[:, 0] * width - (bbox_widths / 2)
#     bbox[:, 1] = bbox[:, 1] * height - (bbox_heights / 2)
#     bbox[:, 2] = bbox[:, 0] + bbox_widths
#     bbox[:, 3] = bbox[:, 1] + bbox_heights
#
#     return bbox


# def yolo_to_voc(ann_dir,
#                 imgfolderpath,
#                 classfilepath,
#                 output_dir=tempfile.gettempdir()):
#     # TODO: @shreeram, make all argument names small
#     """arguments :
#
#     ann_dir : path to the annotations folder
#     imgfolderpath : path to the iamges folder
#     classfilepath : path to the class file (.txt file)
#     output_dir : path where the converted files are to be stored
#     """
#
#     directory = 'ConvertedPascalVOCFiles'
#     dirpath = os.path.join(output_dir, directory)
#     if os.path.exists(dirpath) and os.path.isdir(dirpath):
#         shutil.rmtree(dirpath)
#     os.mkdir(dirpath)
#
#     for file in glob.glob(os.path.join(ann_dir, '*')):
#         if file.endswith('.txt') and file != 'classes.txt':
#
#             split_name = os.path.basename(file)
#             print('Converting', split_name)
#
#             annotation_txt = os.path.splitext(split_name)[0]
#             imgPath = os.path.join(imgfolderpath, annotation_txt + '.jpg')
#
#             image = cv2.imread(imgPath)
#             imageShape = [image.shape[0], image.shape[1], image.shape[2]]
#             imgFolderName = os.path.basename(imgfolderpath)
#             imgFileName = os.path.basename(imgPath)
#
#             writer = PascalVocWriter(
#                 imgFolderName, imgFileName, imageShape, localImgPath=imgPath)
#
#             # Read YOLO files
#             txt_path = os.path.join(ann_dir, split_name)
#             Yolo_reader = YoloReader(txt_path, image, classfilepath)
#             shapes = Yolo_reader.getShapes()
#             num_of_box = len(shapes)
#
#             for i in range(num_of_box):
#                 label = shapes[i][0]
#                 xmin = shapes[i][1][0][0]
#                 ymin = shapes[i][1][0][1]
#                 x_max = shapes[i][1][2][0]
#                 y_max = shapes[i][1][2][1]
#
#                 writer.addBndBox(xmin, ymin, x_max, y_max, label, 0)
#
#         # Write the converted PascalVOC xml files into a new Directory
#             writer.save(
#                 targetFile=os.path.join(dirpath, annotation_txt + '.xml'))
#     return dirpath


# def voc_to_yolo(ann_dir,
#                 imgfolderpath,
#                 debug=False,
#                 output_dir=tempfile.gettempdir()):
#     # TODO: @shreeram, make all argument names small
#     """arguments :
#
#     ann_dir : path to the annotations folder
#     imgfolderpath : path to the iamges folder
#     output_dir : path where the converted files are to be stored
#     """
#
#     directory = 'ConverterdYolotxtFiles'
#     dirpath = os.path.join(output_dir, directory)
#     if os.path.exists(dirpath) and os.path.isdir(dirpath):
#         shutil.rmtree(dirpath)
#     os.mkdir(dirpath)
#     classes = []
#
#     # Search all pascal annotation (xml files) in this folder
#     for file in glob.glob(os.path.join(
#             ann_dir, '*')):  # use glob.glob in place of os.listdir
#         if file.endswith('.xml'):
#
#             split_name = os.path.basename(file)
#             if debug:
#                 print(f'Converting :: {split_name}')
#             annotation_xml = os.path.splitext(split_name)[0]
#
#             imagePath = os.path.join(imgfolderpath, annotation_xml + '.jpg')
#
#             image = cv2.imread(imagePath)
#             imageShape = [image.shape[0], image.shape[1], image.shape[2]]
#             imgFolderName = os.path.basename(imgfolderpath)
#             imgFileName = os.path.basename(imagePath)
#
#             writer = YOLOWriter(
#                 imgFolderName, imgFileName, imageShape, localImgPath=imagePath)
#
#             parser = ET.XMLParser(encoding='utf-8')
#             tree = ET.parse(
#                 os.path.join(ann_dir, str(split_name)), parser=parser)
#             root = tree.getroot()
#             for obj in root.findall('object'):
#                 if str(obj.find('name').text) not in classes:
#                     classes.append(str(obj.find('name').text))
#
#             # Read VOC file
#             filePath = os.path.join(ann_dir, split_name)
#             Voc_Reader = PascalVocReader(filePath, image)
#             shapes = Voc_Reader.getShapes()
#             num_boxes = len(shapes)
#
#             for i in range(num_boxes):
#                 label = classes.index(shapes[i][0])
#                 xmin = shapes[i][1][0][0]
#                 ymin = shapes[i][1][0][1]
#                 x_max = shapes[i][1][2][0]
#                 y_max = shapes[i][1][2][1]
#
#                 writer.addBndBox(xmin, ymin, x_max, y_max, label, 0)
#
#             writer.save(
#                 targetFile=os.path.join(dirpath, annotation_xml + '.txt'))
#
#             classFile = open(os.path.join(output_dir, 'classes.txt'), 'w')
#             for element in classes:
#                 classFile.write(element + '\n')
#     return dirpath


def coco_to_voc(ann_file, output_dir=tempfile.gettempdir()):
    """arguments :

    ann_file : path to the annotations file (.json file)
    output_dir : path where the converted files are to be stored
    """

    coco = COCO(ann_file)
    cats = coco.loadCats(coco.getCatIds())
    cat_idx = {}
    for c in cats:
        cat_idx[c['id']] = c['name']
    for img in coco.imgs:
        catIds = coco.getCatIds()
        annIds = coco.getAnnIds(imgIds=[img], catIds=catIds)
        if len(annIds) > 0:
            img_fname = coco.imgs[img]['file_name']
            image_fname_ls = img_fname.split('.')
            image_fname_ls[-1] = 'xml'
            label_fname = '.'.join(image_fname_ls)
            writer = Writer(img_fname, coco.imgs[img]['width'],
                            coco.imgs[img]['height'])
            anns = coco.loadAnns(annIds)
            for a in anns:
                bbox = a['bbox']
                bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
                bbox = [str(b) for b in bbox]
                catname = cat_idx[a['category_id']]
                writer.addObject(catname, bbox[0], bbox[1], bbox[2], bbox[3])
                writer.save(output_dir + '/' + label_fname)

    return output_dir


def voc_to_coco(ann_dir, debug=False, output_dir=tempfile.gettempdir()):
    """
    ann_dir : path to the annotations folder
    output_dir : path where the converted files are to be stored
    """

    categories = PRE_DEFINED_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    json_dict = {
        'images': [],
        'type': 'instances',
        'annotations': [],
        'categories': []
    }
    image_id = 1
    for file in glob.glob(os.path.join(
            ann_dir, '*')):  # use glob.glob in place of os.listdir
        if file.endswith('xml'):

            split_name = os.path.basename(file)
            if debug:
                print(f'Converting :: {split_name}')
            xml_f = os.path.join(ann_dir, split_name)
            tree = ET.parse(xml_f)
            root = tree.getroot()
            path = root.findtext('path')
            if path is None:
                filename = root.findtext('filename')
            else:
                filename = os.path.basename(path)

            size = get_and_check(root, 'size', 1)
            width = int(get_and_check(size, 'width', 1).text)
            height = int(get_and_check(size, 'height', 1).text)
            image = {
                'file_name': filename,
                'height': height,
                'width': width,
                'id': image_id
            }
            json_dict['images'].append(image)

            for obj in get(root, 'object'):

                category = get_and_check(obj, 'name', 1).text
                if category not in categories:
                    new_id = len(categories)
                    categories[category] = new_id
                category_id = categories[category]
                bndbox = get_and_check(obj, 'bndbox', 1)
                xmin = float(get_and_check(bndbox, 'xmin', 1).text)
                ymin = float(get_and_check(bndbox, 'ymin', 1).text)
                xmax = float(get_and_check(bndbox, 'xmax', 1).text)
                ymax = float(get_and_check(bndbox, 'ymax', 1).text)
                assert (xmax > xmin)
                assert (ymax > ymin)
                o_width = abs(xmax - xmin)
                o_height = abs(ymax - ymin)
                ann = {
                    'area': o_width * o_height,
                    'iscrowd': 0,
                    'image_id': image_id,
                    'bbox': [xmin, ymin, o_width, o_height],
                    'category_id': category_id,
                    'id': bnd_id,
                    'ignore': 0,
                    'segmentation': []
                }
                json_dict['annotations'].append(ann)
                bnd_id = bnd_id + 1

            for cate, cid in categories.items():
                cat = {'supercategory': 'none', 'id': cid, 'name': cate}
                json_dict['categories'].append(cat)
            image_id += 1

            json_file = 'converted_file.json'
            path = os.path.join(output_dir, json_file)
            json_fp = open(path, 'w')
            json_str = json.dumps(json_dict)
            json_fp.write(json_str)
            json_fp.close()

    return (os.path.join(output_dir, json_file))


def voc_to_kitti(ann_dir, output_dir=tempfile.gettempdir()):
    """
    ann_dir : path to the annotations folder
    output_dir : path where the converted files are to be stored
    """

    file_count = 0

    for file in glob.glob(os.path.join(ann_dir, '*')):

        split_name = os.path.basename(file)
        path = os.path.join(ann_dir, split_name)
        if process_file(path, output_dir):
            file_count += 1

    print('Conversion completed. {0} Files are processed'.format(file_count))


def kitti_to_voc(ann_dir, imgfolderpath, output_dir):
    """
    ann_dir : path to the annotations folder
    imgfolderpath : path to the iamges folder
    output_dir : path where the converted files are to be stored
    """

    filter_item = ['DontCare']
    for filename in glob.glob(os.path.join(ann_dir, '*')):

        split_name = os.path.basename(filename)
        annotation_kitti = os.path.splitext(split_name)[0]

        imagePath = os.path.join(imgfolderpath, annotation_kitti + '.jpg')

        img = cv2.imread(imagePath)
        imageShape = [img.shape[0], img.shape[1], img.shape[2]]
        img_name = os.path.basename(imagePath)

        voc_writer = PascalVocWriter(output_dir, img_name, imageShape)
        count = 0
        kitti_ant = open(ann_dir + '/' + split_name, 'r')
        for line in kitti_ant:
            item_list = line.split(' ')
            if item_list[0] not in filter_item:
                count += 1
                voc_writer.addBndBox(
                    (float(item_list[4])), (float(item_list[5])),
                    (float(item_list[6])), (float(item_list[7])), item_list[0],
                    0)

            else:

                img[(float(item_list[5])):(float(item_list[7])),
                    (float(item_list[4])):(float(item_list[6])), :] = 0

        if count > 0:
            voc_writer.save(
                targetFile=os.path.join(output_dir, annotation_kitti + '.xml'))
