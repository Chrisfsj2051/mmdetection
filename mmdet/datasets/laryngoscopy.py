import numpy as np
import joblib
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.datasets import CustomDataset, DATASETS
from mmdet.core import eval_map
from mmdet.utils import get_root_logger


def calculate_confusion_matrix(gt_bboxes, gt_labels, test_results, k, iou_thr,
                               near_gt_bbox):
    assert isinstance(k, int)
    pred_bboxes, confusion_matrix = [], np.zeros(shape=(6, 6))
    for cls_idx, cls_pred in enumerate(test_results):
        if len(cls_pred) == 0:
            continue
        label_array = np.array([cls_idx] * len(cls_pred))[..., None]
        pred_array = np.concatenate([cls_pred, label_array], axis=1)
        pred_bboxes.append(pred_array)
    if len(pred_bboxes):
        pred_bboxes = np.concatenate(pred_bboxes)
        pred_bboxes = pred_bboxes[np.argsort(-pred_bboxes[:, -2])]
    else:
        pred_bboxes = np.empty(shape=(0, 5))

    candidate_label = []
    # iou > 0.5 的所有框， topk; 没有检测出，6
    for i, pred_bbox in enumerate(pred_bboxes):
        pred_cls = pred_bbox[5]
        bbox_overlap = bbox_overlaps(
            pred_bbox[:4][None], gt_bboxes)
        if not near_gt_bbox and i >= k:
            break
        # if bbox_overlap >= iou_thr:
        assert iou_thr <= 0.0
        candidate_label.append(int(pred_cls))

    if len(candidate_label) == 0:
        if len(gt_labels) == 0:
            confusion_matrix[5, 5] += 1
        else:
            confusion_matrix[gt_labels, 5] += 1
    else:
        flag = 0
        for idx in range(min(k, len(candidate_label))):
            assert idx == 0
            if len(gt_labels) == 0:
                if len(candidate_label) == 0:
                    confusion_matrix[5, 5] += 1
                else:
                    confusion_matrix[5, candidate_label[idx]] += 1
                continue
            if len(candidate_label) == 0:
                confusion_matrix[gt_labels, 5] += 1
                continue
            if candidate_label[idx] == gt_labels:
                confusion_matrix[gt_labels, gt_labels] += 1
                flag = 1
                break
        # if flag == 0:
        #     confusion_matrix[gt_labels, candidate_label[0]] += 1

    return confusion_matrix


@DATASETS.register_module()
class Laryngoscopy(CustomDataset):

    def __init__(self, **kwargs):
        super(Laryngoscopy, self).__init__(**kwargs)
        self.CLASSES = ['A', 'B', 'C', 'D', 'E']
        self.best_AP = (0.0, 0)
        self.best_AR = (0.0, 0)
        self.epoch_cnt = 0
        self.class_cnt=[0 for _ in range(6)]

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        return joblib.load(ann_file)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        # if len(ann_info['labels']):
        #     self.class_cnt[ann_info['labels'][0]] += 1
        # else:
        #     self.class_cnt[5] += 1
        # for i in range(6):
        #     print(self.class_cnt[i], end=', ')
        # print('. ')
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """
        if logger is None:
            logger = get_root_logger()
        self.epoch_cnt += 1
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall', 'precision']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = []
        if metric == 'mAP':
            assert False
            assert isinstance(iou_thr, float)
            mean_ap, _ = eval_map(
                results,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=iou_thr,
                dataset=self.CLASSES,
                logger=logger)
            eval_results['mAP'] = mean_ap
        elif metric == 'recall':
            # topks = (1, 3, 5)
            topks = (1,)
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            gt_labels = [ann['labels'] for ann in annotations]
            confusion_matrixs = [np.zeros(shape=(6, 6)) for _ in range(3)]
            for (gt_bbox, gt_label, test_result) in zip(
                    gt_bboxes, gt_labels, results):
                for i in range(len(topks)):
                    # cm = calculate_confusion_matrix(
                    #     gt_bbox, gt_label, test_result, topks[i],
                    #     iou_thr=0.5, near_gt_bbox=False)
                    cm = calculate_confusion_matrix(
                        gt_bbox, gt_label, test_result, topks[i],
                        iou_thr=-1.0, near_gt_bbox=False)
                    confusion_matrixs[i] += cm

            for i in range(len(topks)):
                cls_res, cm = [], confusion_matrixs[i]
                eval_res = {'k': topks[i], 'cm': cm,
                            'precision': [], 'recall': []}
                for cls in range(6):
                    TP = cm[cls, cls]
                    FN = cm[cls].sum() - TP
                    FP = cm[:, cls].sum() - TP
                    eval_res['precision'].append(TP / (1e-6 + TP + FP))
                    eval_res['recall'].append(TP / (1e-6 + TP + FN))

                eval_res['precision'].append(round(np.array(eval_res['precision']).mean(), 4))
                eval_res['recall'].append(round(np.array(eval_res['recall']).mean(), 4))
                eval_results.append(eval_res)
                star_num = 56
                if i == 0:
                    logger.info('')
                    logger.info('*' * star_num)
                    logger.info("| %5s | %-20s | %10s | %8s *" % ('Top-k', 'Category', 'Precision', 'Recall'))
                    logger.info('*' * star_num)

                CLASSES = ('Laryngeal Carcinoma', 'Cord Cyst', 'Nodules',
                           'Polyps', 'Leukoplakia', 'Normal', 'Average')
                for idx in range(7):
                    # print("| %5s | %-20s | %10s | %7s |" % (
                    #     f'top-{topks[i]}', CLASSES[idx],
                    #     "%.2f %%" % (eval_res['precision'][idx] * 100),
                    #     "%.2f %%" % (eval_res['recall'][idx] * 100),
                    # ))
                    logger.info("| %5s | %-20s | %10s | %8s |" % (
                        f'top-{topks[i]}', CLASSES[idx],
                        "%.2f %%" % (eval_res['precision'][idx] * 100),
                        "%.2f %%" % (eval_res['recall'][idx] * 100),
                    ))

                logger.info('*' * star_num)

            for idx in range(len(topks)):
                logger.info(f'\nConfusion Matrix for top-{topks[idx]}:\n'
                            + str(eval_results[idx]['cm']).replace('.', ','))

        if self.best_AP[0] < eval_results[0]['precision'][-1]:
            self.best_AP = (eval_results[0]['precision'][-1], self.epoch_cnt)
        if self.best_AR[0] < eval_results[0]['recall'][-1]:
            self.best_AR = (eval_results[0]['recall'][-1], self.epoch_cnt)

        return {f'AP-top{topks[0]}': eval_results[0]['precision'][-1],
                f'AR-top{topks[0]}': eval_results[0]['recall'][-1],
                f'AP-best': self.best_AP,
                f'AR-best': self.best_AR}

    # top-k个找到了，那就算找到了
