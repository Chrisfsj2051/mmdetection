from .faster_rcnn import FasterRCNN
from ..builder import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class LaryngoscopyFasterRCNN(FasterRCNN):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        return super(LaryngoscopyFasterRCNN, self).simple_test(
            img, img_metas, proposals, rescale=True
        )
