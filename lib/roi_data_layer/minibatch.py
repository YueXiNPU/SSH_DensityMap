# ------------------------------------------------------------------------------------------------
# This file is a modified version of https://github.com/rbgirshick/py-faster-rcnn by Ross Girshick
# Modified by Mahyar Najibi
# ------------------------------------------------------------------------------------------------
import cv2
import numpy as np
import numpy.random as npr
from utils.get_config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob
import h5py
from matplotlib import pyplot as plt
from matplotlib import cm as CM


def get_minibatch(roidb):
    """Return the mini-batch for training"""
    num_images = len(roidb)
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)

    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

    blobs = {'data': im_blob}

    assert len(im_scales) == 1, "Single batch only"
    assert len(roidb) == 1, "Single batch only"
    # gt boxes: (x1, y1, x2, y2, cls)
    gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
    gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
    gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
    blobs['gt_boxes'] = gt_boxes
    blobs['im_info'] = np.array(
        [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
        dtype=np.float32)

    #ground truth of density map by Yue,
    #blobs['gt_densityMap'] = np.array([0,0,0,0])
    densityMap_gt_blob, densityMap_gt_scales = _get_gt_densityMap_blob(roidb, random_scale_inds)
    blobs['gt_densityMap'] = densityMap_gt_blob
    #print(blobs['gt_densityMap'])

    return blobs

def prep_gt_densityMap_for_blob(im, target_size, max_size):
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    if cfg.TRAIN.ORIG_SIZE:
        im_scale = 1
    else:
        im_scale = float(target_size) / float(im_size_min)

    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    return im, im_scale

def gt_densityMap_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 1),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1],0] = im

    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob




def _get_gt_densityMap_blob(roidb, scale_inds):
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in xrange(num_images):

        gt_densityMap_file = h5py.File(roidb[i]['densityMap'], 'r')
        im = np.asarray(gt_densityMap_file['density'])
        #plt.imshow(im, cmap=CM.jet)
        #plt.show()


        if roidb[i]['flipped']:
            im = im[:, ::-1]

        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_gt_densityMap_for_blob(im, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)

    blob = gt_densityMap_list_to_blob(processed_ims)
    return blob, im_scales

def _get_image_blob(roidb, scale_inds):
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in xrange(num_images):
        im = cv2.imread(roidb[i]['image'])

        if roidb[i]['flipped']:
            im = im[:, ::-1, :]

        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)

    blob = im_list_to_blob(processed_ims)
    return blob, im_scales


def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois = im_rois * im_scale_factor
    return rois

