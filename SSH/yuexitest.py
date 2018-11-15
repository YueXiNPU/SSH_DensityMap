import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io as scio

def vis_gt(im, bboxes, plt_name='output', ext='.png', visualization_folder=None):
    """
    A function to visualize the detections
    :param im: The image
    :param bboxes: ground truth
    :param plt_name: The name of the plot
    :param ext: The save extension (if visualization_folder is not None)
    :param visualization_folder: The folder to save the results
    :param thresh: The detections with a score less than thresh are not visualized
    """

    fig, ax = plt.subplots(figsize=(12, 12))


    ax.imshow(im, aspect='equal')
    if bboxes.shape[0] != 0:

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, :]
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor=(0, bbox[4], 0), linewidth=3)
            )

    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    if visualization_folder is not None:
        if not os.path.exists(visualization_folder):
            os.makedirs(visualization_folder)
        plt_name += ext
        plt.savefig(os.path.join(visualization_folder, plt_name), bbox_inches='tight')
        print('Saved {}'.format(os.path.join(visualization_folder, plt_name)))
    else:
        print('Visualizing {}!'.format(plt_name))
        plt.show()
    plt.clf()
    plt.cla()

if __name__ == '__main__':
    imdb = scio.loadmat("../data/datasets/wider/wider_face_split/wider_face_val.mat")
    print(imdb.items)
    print("well done!")
