"""Summary."""
import numpy as np
import os
import xml.etree.ElementTree as ET
from glob import glob
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm


def plot_coco_instance(image_array, bboxes, labels):
    """Summary.

    Args:
        image_array (np.ndarray): Description
        bboxes (np.ndarray): (N,4) normalized bboxes in (x1,y1,width,height)
        labels (np.ndarray): Description
    """
    assert isinstance(
        bboxes, np.ndarray) and bboxes.ndim == 2 and bboxes.shape[
            1] == 4, f'bboxes should be numpy of dimension (Nx4), got\
        {bboxes.shape}'

    bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] + bboxes[:, 1]

    return plot_voc_instance(image_array, bboxes, labels)


def annotate_coco_dataset(images_dir,
                          annotation_file,
                          result_dir,
                          ignore_missing=False):
    coco = COCO(annotation_file)

    image_ids = coco.getImgIds()

    for img_id in tqdm(image_ids):

        annotation_id = coco.getAnnIds(img_id)
        img_info = coco.loadImgs(img_id)
        assert len(img_info) == 1

        img_name = img_info[0]['file_name']
        annotations = coco.loadAnns(annotation_id)

        bboxes = []
        labels = []

        for annotation in annotations:
            bboxes.append(annotation['bbox'])
            labels.append(annotation['category_id'])

        if bboxes == []:
            print(f'No annotations found for {img_name}')
            continue

        bboxes = np.array(bboxes, dtype='float32')
        labels = np.array(labels)

        imagefile = os.path.join(images_dir, img_name)
        filename, extension = os.path.splitext(imagefile)

        # comment/uncomment below to plot and save
        with open(imagefile, 'rb') as im_handle:
            pil_image = Image.open(im_handle)
            image_array = np.array(pil_image)
        burnt_image = plot_coco_instance(image_array, bboxes, labels)
        save_path = filename.replace(images_dir, result_dir) + '_annotated.jpg'

        burnt_image.save(save_path)


def plot_voc_instance(image_array, bboxes, labels):
    """Summary.

    Args:
        image_array (np.ndarray): Description
        bboxes (np.ndarray): (N,4) bboxes in (x1,y1,x2,y2)
        labels (np.ndarray): Description

    Returns:
        TYPE: Description
    """
    assert isinstance(image_array, np.ndarray), 'image array should be numpy'
    assert isinstance(bboxes,
                      np.ndarray) and bboxes.ndim == 2 and bboxes.shape[
                          1] == 4, 'bboxes should be numpy of dimension (Nx4)'
    assert isinstance(labels,
                      np.ndarray), 'labels should be numpy of dimesnion (N,)'
    scores = np.ones_like(labels, dtype='float32')

    return annotate_image(image_array, bboxes, scores, labels)  # noqa: F821


def annotate_voc_dataset(images_dir,
                         annotation_dir,
                         result_dir,
                         ignore_missing=False):

    for imagefile in tqdm(glob(os.path.join(images_dir, '*.*'))):

        with open(imagefile, 'rb') as im_handle:
            pil_image = Image.open(im_handle)
            image_array = np.array(pil_image)

        filename, extension = os.path.splitext(imagefile)

        anno_path = filename.replace(images_dir, annotation_dir) + '.xml'
        file = ET.parse(anno_path)

        xmin, ymin, xmax, ymax, label_name = [], [], [], [], []
        for anno in file.iter('object'):
            xmin.append(float(anno.find('bndbox').find('xmin').text))
            ymin.append(float(anno.find('bndbox').find('ymin').text))
            xmax.append(float(anno.find('bndbox').find('xmax').text))
            ymax.append(float(anno.find('bndbox').find('ymax').text))
            label_name.append(anno.find('name').text)

        # image_array = np.array(pil_image)
        bboxes = np.array([xmin, ymin, xmax, ymax]).T
        labels = np.array(label_name)

        burnt_image = plot_voc_instance(image_array, bboxes, labels)
        save_path = filename.replace(images_dir, result_dir) + '_annotated.jpg'

        burnt_image.save(save_path)


def plot_yolo_instance(image_array, bboxes, labels):
    """Summary.

    Args:
        image_array (np.ndarray): Description
        bboxes (np.ndarray): Description
        labels (np.ndarray): Description
    """
    assert isinstance(bboxes,
                      np.ndarray) and bboxes.ndim == 2 and bboxes.shape[
                          1] == 4, 'bboxes should be numpy of dimension (Nx4)'
    height, width, c = image_array.shape

    bbox_widths = bboxes[:, 2] * width
    bbox_heights = bboxes[:, 3] * height

    bboxes[:, 0] = bboxes[:, 0] * width - (bbox_widths / 2)
    bboxes[:, 1] = bboxes[:, 1] * height - (bbox_heights / 2)
    bboxes[:, 2] = bboxes[:, 0] + bbox_widths
    bboxes[:, 3] = bboxes[:, 1] + bbox_heights

    return plot_voc_instance(image_array, bboxes, labels)


def annotate_yolo_dataset(images_dir,
                          annotation_dir,
                          result_dir,
                          ignore_missing=False):

    for imagefile in tqdm(glob(os.path.join(images_dir, '*.*'))):

        with open(imagefile, 'rb') as im_handle:
            pil_image = Image.open(im_handle)
            image_array = np.array(pil_image)

        filename, extension = os.path.splitext(imagefile)

        anno_path = filename.replace(images_dir, annotation_dir) + '.txt'
        # file=ET.parse(anno_path)

        label_name, x_center, y_center = [], [], []
        relative_width, relative_height = [], []

        with open(anno_path, 'r') as f:

            data = f.readline().rstrip()
            while data:
                label, x1, y1, w, h = data.split(' ')

                x_center.append(x1)
                y_center.append(y1)
                relative_width.append(w)
                relative_height.append(h)
                label_name.append(int(label))
                data = f.readline().rstrip()

        # image_array = np.array(pil_image)
        bboxes = np.array(
            [x_center, y_center, relative_width, relative_height],
            dtype='float32').T
        labels = np.array(label_name)

        burnt_image = plot_yolo_instance(image_array, bboxes, labels)
        save_path = filename.replace(images_dir, result_dir) + '_annotated.jpg'

        burnt_image.save(save_path)


def plot_kitti_instance(image_array, bboxes, labels):
    """Summary.

    Args:
        image_array (np.ndarray): Description
        bboxes (np.ndarray): (N,4) bboxes in (y1, x1, y2, x2)
        labels (np.ndarray): Description
    """
    bboxes = np.transpose(bboxes, (1, 0, 3, 2))

    return plot_voc_instance(image_array, bboxes, labels)
