import numpy as np
import joblib
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.datasets import CustomDataset, DATASETS
from mmdet.core import eval_map
from mmdet.utils import get_root_logger
import time
import random


@DATASETS.register_module()
class Laryngoscopy(CustomDataset):

    def __init__(self, class_equal=False, normal_thr=0.3,
                 save_roc_path=None, **kwargs):
        super(Laryngoscopy, self).__init__(**kwargs)
        self.CLASSES = ('Laryngeal Carcinoma', 'Pre Cancer', 'Cyst   ',
                        'Pol&Nod', 'Normal')
        self.best_AP = (-1, 0)
        self.best_AR = (-1, 0)
        self.best_Ave = (-1, -1, 0)
        self.epoch_cnt = 0
        self.class_cnt = [0 for _ in range(len(self.CLASSES))]
        self.best_test_result = None
        self.class_equal = class_equal
        self.normal_thr = normal_thr
        self.dump_name = time.strftime(
            f'{save_roc_path}/roc_test_result.pkl',
            time.localtime())
        if class_equal:
            self.class_inds = [[] for _ in range(len(self.CLASSES))]
            for (i, ann) in enumerate(self.data_infos):
                category = (ann['ann']['labels'][0]
                            if len(ann['ann']['labels']) else
                            len(self.CLASSES) - 1)
                self.class_inds[category].append(i)

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        return joblib.load(ann_file)

    def __getitem__(self, idx):
        if self.test_mode or not self.class_equal:
            return super(Laryngoscopy, self).__getitem__(idx)
        class_idx = random.choice(range(len(self.CLASSES)))
        while True:
            idx = random.choice(self.class_inds[class_idx])
            data = self.prepare_train_img(idx)
            if data is None:
                idx = random.choice(self.class_inds[class_idx])
                continue
            return data

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

        image_info = self.data_infos
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        pred_result = []
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
            topks = (1,)
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            gt_labels = [ann['labels'] for ann in annotations]
            confusion_matrixs = np.zeros(shape=(5, 5))

            for (i, (gt_bbox, gt_label, test_result)) in \
                    enumerate(zip(gt_bboxes, gt_labels, results)):
                cm, pred_res = self.calculate_confusion_matrix(
                    gt_label, test_result, image_info[i])
                confusion_matrixs += cm
                pred_result.append(pred_res)

            cls_res, cm = [], confusion_matrixs
            eval_res = {'k': 1, 'cm': cm,
                        'precision': [], 'recall': []}
            for cls in range(len(self.CLASSES)):
                TP = cm[cls, cls]
                FN = cm[cls].sum() - TP
                FP = cm[:, cls].sum() - TP
                eval_res['precision'].append(TP / (1e-6 + TP + FP))
                eval_res['recall'].append(TP / (1e-6 + TP + FN))

            eval_res['precision'].append(round(np.array(eval_res['precision']).mean(), 4))
            eval_res['recall'].append(round(np.array(eval_res['recall']).mean(), 4))
            eval_results.append(eval_res)
            star_num = 56

            logger.info('')
            logger.info('*' * star_num)
            logger.info("| %5s | %-20s | %10s | %8s *" % ('Top-k', 'Category', 'Precision', 'Recall'))
            logger.info('*' * star_num)
            CLASSES = self.CLASSES + ('Average',)
            for idx in range(len(CLASSES)):
                logger.info("| %5s | %-20s | %10s | %8s |" % (
                    f'top-{1}', CLASSES[idx],
                    "%.2f %%" % (eval_res['precision'][idx] * 100),
                    "%.2f %%" % (eval_res['recall'][idx] * 100),
                ))

            logger.info('*' * star_num)

        for idx in range(len(topks)):
            logger.info(f'\nConfusion Matrix for top-{topks[idx]}:\n'
                        + str(eval_results[idx]['cm']).replace('.', ','))

        Prec, Rec = eval_results[0]['precision'][-1], eval_results[0]['recall'][-1]
        best_Ave = self.best_Ave[0] + self.best_Ave[1]
        # if self.best_AP[0] < Prec:
        #     self.best_AP = (Prec, self.epoch_cnt)
        # if self.best_AR[0] < Rec:
        #     self.best_AR = (Rec, self.epoch_cnt)
        if best_Ave < Prec + Rec:
            self.best_Ave = (Prec, Rec, self.epoch_cnt)
            self.best_test_result = pred_result

        joblib.dump(self.best_test_result, self.dump_name)

        return {f'AP-top{topks[0]}': eval_results[0]['precision'][-1],
                f'AR-top{topks[0]}': eval_results[0]['recall'][-1],
                # f'AP-best': self.best_AP,
                # f'AR-best': self.best_AR,
                f'Ave-best': self.best_Ave, }

    def calculate_confusion_matrix(self, gt_labels, test_results, image_info):
        total_class_num = len(self.CLASSES)
        bg_inds = total_class_num - 1
        pred_bboxes = []
        confusion_matrix = np.zeros(shape=(total_class_num,
                                           total_class_num))

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
            pred_bboxes = np.empty(shape=(0, total_class_num))

        gt_label = gt_labels[0] if len(gt_labels) else bg_inds
        pred_result = dict(
            pred_score=pred_bboxes[0][4],
            gt_label=gt_label,
            filename=image_info['filename']
        )
        pred_label = (bg_inds
                      if pred_bboxes[0][4] < self.normal_thr
                      else int(pred_bboxes[0][5]))
        pred_result['pred_label'] = pred_label
        confusion_matrix[gt_label, pred_label] += 1

        return confusion_matrix, pred_result
