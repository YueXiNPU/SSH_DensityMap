import caffe
import numpy as np


#
# layer {
#   name: "resizelayer"
#   type: "Python"
#   top: "gt_resize_densityMap"
#   bottom: "ssh_densityMap"
#   bottom: "gt_densityMap"
#   python_param {
#     module: "SSH.layers.resize_layer"
#     layer: "ResizeLayer"
#   }
# }


class ResizeLayer(caffe.Layer):
    def setup(self, bottom, top):
        height, width = bottom[0].data.shape[-2:]
        top[0].reshape(1, 1, height, width)
        pass

    def forward(self, bottom, top):
        height, width = bottom[0].data.shape[-2:]
        # gt_densityMap = bottom[1].data


        # Reshape net's input blobs
        top[0].reshape(*(bottom[0].shape))

        # resize densityMap into specific shape
        gt_densityMap = bottom[1].data
        gt_reiszed_densityMap = caffe.io.resize(gt_densityMap,(1,1,height, width))



        # Copy data into net's input blobs
        top[0].data[...] = gt_reiszed_densityMap
        #top[0].data[...] = bottom[1].data.astype(np.float32, copy=False)


        #top[0].data[...] = gt_reiszed_densityMap

        pass

    def reshape(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass
