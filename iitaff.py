# %matplotlib inline

import os
import sys
import random
import math
import re
import time
import numpy as np
# import cv2
import matplotlib
import matplotlib.pyplot as plt

from config import Config
import utils
import model as modellib
import visualize
from model import log
from PIL import Image
from scipy.ndimage.interpolation import zoom # as SCIPY_zoom


# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


## -------------------------- Configurations ----------------------
class IITAFFConfig(Config):
    """Configuration for training on the IIT-AFF dataset.
    Derives from the base Config class and overrides values specific
    to the dataset.
    """
    # Give the configuration a recognizable name
    NAME = "IITAFF"

    # Train on 1 GPU and 2 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 2 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 9  # background + 9 affordances (or 10 object classes)

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    # TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    # STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    # VALIDATION_STEPS = 5

config = IITAFFConfig()
config.display()

## -------------------------- Notebook Preferences ----------------------
def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

class IITAFFDataset(utils.Dataset):
    """ Load IIT AFF Dataset.
    """
    def __init__(self):
        super(IITAFFDataset, self).__init__()
        self.name = 'IITAFF'
        self.W = 512
        self.H = 512

    def load_iitaff( self, dataset_dir = '/home/niu/Liang_Niu3/IIT_Affordances_2017', subset = 'train'):
        """ Load a subset of IIT AFF Dataset
        dataset_dir: The root directory of the IIT-AFF dataset.
        subset: What to load (train, val, train_val, test)
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        assert subset in ('train', 'val', 'test')

        print("Dataset_Dir:", dataset_dir)

        self.dataset_dir = dataset_dir
        self.image_dir = os.path.join(dataset_dir, 'rgb')
        self.aff_dir = os.path.join(dataset_dir, 'affordances_labels')
        # class_names = ['background', 'contain', 'cut', 'display', 'engine',
        class_names = ['contain', 'cut', 'display', 'engine',
                       'grasp', 'hit', 'pound', 'support', 'w-grasp']

        # Add classes
        for id, name in enumerate(class_names):
            self.add_class(self.name, id+1, name)

        # Load all files into list
        def get_file_list():
            """ Get name list for train/val/test data.
            Statistics: 4417 train, 1767 val, 2651 test. In total: 8835.
            TODO: Later we may want to divide the train_val in a defferent way.
            Like only take first 500 in val list as val, others will be added into train list.
            """
            with open(dataset_dir+'/train_and_val.txt') as f:
                train_val = [ line.rstrip('\n') for line in f]
            with open(dataset_dir+'/val.txt') as f:
                val       = [ line.rstrip('\n') for line in f]
            with open(dataset_dir+'/test.txt') as f:
                test      = [ line.rstrip('\n') for line in f]
            train = list(set(train_val) - set(val))
            return train, val, test

        train_names, val_names, test_names = get_file_list()
        self.all_names = train_names + val_names + test_names
        self.id_base = {'train':0, 'val':len(train_names), 'test':len(train_names)+len(val_names)}[subset]
        subset_dict = {'train':train_names, 'val':val_names, 'test':test_names}
        self.length = len(subset_dict[subset])
        self.id2name = subset_dict[subset]

        # Add images
        for idx, rgb_name in enumerate(subset_dict[subset]):
            rgb_path = os.path.join(self.image_dir, rgb_name)
            img_id = idx # + self.id_base
            if idx%1000==0:
                print(idx, "has been loaded.")
            # w, h = self.loadWH(rgb_path)
            self.add_image(
                self.name, image_id=img_id,
                path=rgb_path,
                width=self.W,
                height=self.H,
            )
        print("Load IIT-AFF done. All ", self.length, " images are loaded.")

    def loadAnn(self, rgb_name): # TODO: is this one correct?
        '''
        This function load and return the affordance mask of an rgb image
        :param rgb_name:
        :return:
        '''
        assert type(rgb_name)==str
        rgb2txt = lambda rgb: os.path.splitext(rgb)[0] + '.txt' # turn extension from jpg to txt
        aff_path = os.path.join(self.aff_dir, rgb2txt(rgb_name))
        # w, h = self.loadWH(os.path.join(self.image_dir, rgb_name))
        ann = np.loadtxt(aff_path, dtype=np.int32)
        w, h = ann.shape
        # resize the annotation into a fixed resolution
        w, h = float(w), float(h)
        return zoom(ann, (self.W/w, self.H/h), order=0)

    # def loadWH(self, rgb_path): # TODO: may become the bottleneck
    #     '''
    #     Return the width and height of an rgb image according to its absolute path.
    #     :param rgb_path:
    #     :return:
    #     '''
    #     im = Image.open(rgb_path)
    #     return im.size  # (width,height) tuple

    def annToMask(self, ann):
        '''
        Turn [height, width] annotations of affordance into [height, width, instances] binary mask.
        :param ann: return value from self.loadAnns
        :return:
        '''
        # filter all non zero number from matrix and put into a set
        ann_uni = np.unique(ann)
        # make binary mask for every number and stack them
        masks = []
        class_ids = []
        for n in ann_uni:
            if n:
                class_ids.append(n)
                masks.append(ann==n)
        # TODO: fix bug if the masks are simply all zero.
        return np.stack(masks, axis=2), np.array(class_ids)


    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance, BG not included.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # separate annotation into [height, width, instance count].
        # assert type(image_id)==int, 'The parameter image_id must be int type.'
        rgb_name = self.id2name[image_id]
        return self.annToMask(self.loadAnn(rgb_name))

    def random_image_ids(self, n=1):
        '''
        TODO: make it real random
        Return n random images' ids.
        :param n: how many image ids to return, default 1.
        :return: a list of n image_ids.
        '''
        # return self.id_base + 42
        random.seed(42)
        return random.choices(range(self.length), k=n)


def evaluate_iitaff(model, dataset, config, do_visualize=False):
    '''
    Two steps.
    :param model:
    :param dataset:
    :return:
    '''

    # First test on some random images
    image_id = dataset.random_image_ids()[0]
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset, config, image_id)
    if do_visualize:
        print("Ground Truth for this image:")
        visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                    dataset.class_names, figsize=(8, 8))
    result = model.detect([original_image], verbose=1)
    r = result[0]
    if do_visualize:
        print("Predict Result for this image:")
        visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                dataset.class_names, r['scores'], ax=get_ax())
    else:
        # TODO: save the predict result
        pass

    # Then, test on all images and calculate the score
    pass
