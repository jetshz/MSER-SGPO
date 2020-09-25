import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image
import os.path
import scipy.misc
from xml.dom.minidom import parse
import xml.dom.minidom
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from scipy.misc import imresize
import skimage
from pycocotools.coco import COCO

IMG_FOLDER_NAME = "JPEGImages"
ANNOT_FOLDER_NAME = "Annotations"

CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']

#CAT_LIST_COCO = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic', 'light', 'fire', 'hydrant', 'stop', 'sign', 'parking', 'meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports', 'ball', 'kite', 'baseball', 'bat', 'baseball', 'glove', 'skateboard', 'surfboard', 'tennis', 'racket', 'bottle', 'wine', 'glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot', 'dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted', 'plant', 'bed', 'dining', 'table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell', 'phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy', 'bear', 'hair', 'drier', 'toothbrush']

CAT_NAME_TO_NUM = dict(zip(CAT_LIST,range(len(CAT_LIST))))

def load_image_label_from_xml(img_name, voc12_root):
    from xml.dom import minidom

    el_list = minidom.parse(os.path.join(voc12_root, ANNOT_FOLDER_NAME,img_name + '.xml')).getElementsByTagName('name')

    multi_cls_lab = np.zeros((20), np.float32)

    for el in el_list:
        cat_name = el.firstChild.data
        if cat_name in CAT_LIST:
            cat_num = CAT_NAME_TO_NUM[cat_name]
            multi_cls_lab[cat_num] = 1.0

    return multi_cls_lab

def load_image_label_list_from_xml(img_name_list, voc12_root):

    return [load_image_label_from_xml(img_name, voc12_root) for img_name in img_name_list]

def get_img_path(img_name, voc12_root):
    return os.path.join(voc12_root, IMG_FOLDER_NAME, img_name + '.jpg')

def load_img_name_list(dataset_path):
    img_gt_name_list = open(dataset_path).read().splitlines()

    return img_gt_name_list

class VOC12ImageDataset(Dataset):

    def __init__(self, img_name_list_path, voc12_root, transform = None):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root
        self.transform = transform

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]

        img = PIL.Image.open(get_img_path(name, self.voc12_root)).convert("RGB")

        size = img.size

        if self.transform:
            img = self.transform(img)

        return name, img, size

class VOC12ClsDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None):
        super().__init__(img_name_list_path, voc12_root, transform)
        self.label_list = load_image_label_list_from_xml(self.img_name_list, self.voc12_root)

    def __getitem__(self, idx):
        name, img, size = super().__getitem__(idx)

        label = torch.from_numpy(self.label_list[idx])

        return name, img, label, size

class VOC12CAMDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None):
        super().__init__(img_name_list_path, voc12_root, transform)
        self.label_list = load_image_label_list_from_xml(self.img_name_list, self.voc12_root)

    def __getitem__(self, idx):
        name, img, size = super().__getitem__(idx)

        label = torch.from_numpy(self.label_list[idx])

        c_map = torch.LongTensor(imresize(np.array(PIL.Image.open('/home/zxforchid/Hu_Z/pychram_PRM/cam_trainval/cam_l/%s.png' % (name))), (112, 112), interp='nearest'))

        return name, img, label, size, c_map

class VOC12ClsDatasetMSF(VOC12ClsDataset):

    def __init__(self, img_name_list_path, voc12_root, scales, inter_transform=None, unit=1):
        super().__init__(img_name_list_path, voc12_root, transform=None)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def __getitem__(self, idx):
        name, img, label, = super().__getitem__(idx)

        rounded_size = (int(round(img.size[0]/self.unit)*self.unit), int(round(img.size[1]/self.unit)*self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0]*s),
                           round(rounded_size[1]*s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.inter_transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.inter_transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(np.flip(ms_img_list[i], -1).copy())

        return name, msf_img_list, label

def load_image_label_list_from_xml_coco(img_name_list, voc12_root):

    return [load_image_label_from_xml(img_name, voc12_root) for img_name in img_name_list]


class SimpleCoCoDataset(Dataset):
    def __init__(self, rootdir, set_name='train2017', transform=None):
        self.rootdir, self.set_name = rootdir, set_name
        self.transform = transform
        self.coco = COCO(os.path.join(self.rootdir, 'annotations', 'instances_'
                                      + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()
        self.load_classes()

    def load_classes(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        # coco ids is not from 1, and not continue
        # make a new index from 0 to 79, continuely

        # classes:             {names:      new_index}
        # coco_labels:         {new_index:  coco_index}
        # coco_labels_inverse: {coco_index: new_index}
        self.classes, self.coco_labels, self.coco_labels_inverse = {}, {}, {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # labels:              {new_index:  names}
        self.labels = {}
        for k, v in self.classes.items():
            self.labels[v] = k

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        img = self.load_image(index)
        ann = self.load_anns(index)
        size = np.array(img).shape[:2]
        image_info = self.coco.loadImgs(self.image_ids[index])[0]
        imgpath = os.path.join(self.rootdir, '', self.set_name,
                               image_info['file_name'])

        if self.transform:
            t_img = self.transform(img)
        sample = {'img': t_img, 'ann': ann, 'size': size, 'path': imgpath, 'name': image_info['file_name']}
        return sample

    def load_image(self, index):
        image_info = self.coco.loadImgs(self.image_ids[index])[0]
        imgpath = os.path.join(self.rootdir, '', self.set_name,
                               image_info['file_name'])

        img = PIL.Image.open(imgpath).convert("RGB")

        return img

    def load_anns(self, index):
        annotation_ids = self.coco.getAnnIds(self.image_ids[index], iscrowd=False)
        # anns is num_anns x 5, (x1, x2, y1, y2, new_idx)
        anns = np.zeros((80), np.float32)

        # skip the image without annoations
        if len(annotation_ids) == 0:
            return anns

        coco_anns = self.coco.loadAnns(annotation_ids)
        for a in coco_anns:
            temp = self.coco_labels_inverse[a['category_id']]
            anns[temp] = 1.0

        return anns

    def image_aspect_ratio(self, index):
        image = self.coco.loadImgs(self.image_ids[index])[0]
        return float(image['width']) / float(image['height'])