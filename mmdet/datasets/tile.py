import re
from collections import OrderedDict, defaultdict
from copy import deepcopy
import os
import mmcv
import numpy as np
import torch
from mmdet.core import eval_map, multiclass_nms
from mmdet.datasets import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.utils import get_root_logger


@DATASETS.register_module()
class TileDefectDataset(CustomDataset):
    CLASSES = ('edge', 'corner', 'white_point', 'light_block', 'dark_block',
               'halo')

    def __init__(self,
                 *args,
                 score_threshold=0.0,
                 nms_cfg=dict(iou_thr=0.01),
                 analysis=False,
                 visualize=False,
                 visualize_dump_path='./det_vis/',
                 **kwargs):

        super(TileDefectDataset, self).__init__(*args, **kwargs)
        self.nms_cfg = nms_cfg
        if not isinstance(score_threshold, float):
            raise NotImplementedError
        self.score_threshold = score_threshold
        if analysis:
            self.analysis_data()
        if visualize:
            self.visualize_gt(visualize_dump_path)
        get_root_logger().info(str(self))

    def __repr__(self):
        """Pretty print the class, including the total number of images and
        instance counts of difference classes.
        """
        dataset_type = 'Test' if self.test_mode else 'Train'
        fmt_str = f'\n{dataset_type} dataset, number of instances: {len(self)}'
        if self.CLASSES is None:
            return fmt_str
        counts = defaultdict(int)
        # background is the last index from mmdet-v2.0
        all_classes = list(self.CLASSES) + ['background']
        for i in range(len(self)):
            data_info = self.get_ann_info(i)
            if len(data_info['labels']) == 0:
                counts['background'] += 1
            else:
                for label in data_info['labels']:
                    counts[all_classes[label]] += 1
        # print instance count for each class
        for i, cls in enumerate(all_classes):
            if i % 5 == 0:
                fmt_str += '\n'
            count = counts[cls]
            cls = cls[:10]
            fmt_str += f'{cls:>10s}[{i:4d}]: {count:6d}; '

        fmt_str += '\n'
        return fmt_str

    def load_meta_annotations(self, ann_file):
        """load official annotation and re-organize it"""
        json_anns = mmcv.load(ann_file)
        data_infos = []
        filename_map = {}
        for img_info in json_anns:
            filename = img_info['name']
            if filename not in filename_map.keys():
                filename_map[filename] = len(data_infos)
                data_infos.append({})
            data_info = data_infos[filename_map[filename]]
            if len(data_info.keys()):
                assert img_info['image_height'] == data_info['height']
                assert img_info['image_width'] == data_info['width']
            else:
                data_info['height'] = img_info['image_height']
                data_info['width'] = img_info['image_width']
                data_info['filename'] = filename
                data_info['ann'] = dict(labels=[], bboxes=[])

            assert (isinstance(img_info['category'], int)
                    and img_info['category'] != 0)
            # since no images belongs to background cetegory,
            # here we use c - 1 to make it start from 0.
            data_info['ann']['labels'].append(img_info['category'] - 1)
            data_info['ann']['bboxes'].append(img_info['bbox'])

        for data_info in data_infos:
            data_info['ann']['labels'] = np.array(
                data_info['ann']['labels']).astype(np.long)
            data_info['ann']['bboxes'] = np.array(
                data_info['ann']['bboxes']).astype(np.float32)

        return data_infos

    def decode_filename(self, img_name):
        """decode patch location from image name"""
        s = re.findall('lw.*@', img_name)[0]
        return [int(x) for x in re.findall(r'\d+', s)]

    def load_empty_annotations(self):
        """
        Since test set doesn't have annotation file, here
        we initialize a pseudo meta_data_infos for it.
        """
        pure_filename = list(
            set([f.split('@')[-1] for f in self.image_name_list]))
        meta_data_infos = []
        for filename in pure_filename:
            meta_data_infos.append(
                dict(
                    filename=filename,
                    height=200,
                    width=300,
                    ann=dict(
                        labels=np.empty((0,)).astype(np.long),
                        bboxes=np.empty((0, 4)).astype(np.float32))))
        return meta_data_infos

    def load_annotations(self, ann_file):
        """Load Annotations"""
        self.image_name_list = os.listdir(self.img_prefix)
        self.meta_data_infos = (
            self.load_meta_annotations(ann_file)
            if ann_file is not None else self.load_empty_annotations())
        self.filename_map = dict()
        for (i, item) in enumerate(self.meta_data_infos):
            self.filename_map[item['filename']] = i
        data_infos = []
        for img_name in self.image_name_list:
            pure_name = img_name.split('@')[-1]
            # images of val-dev and train-dev are in the same folder
            if pure_name not in self.filename_map.keys():
                continue
            data_info = deepcopy(
                self.meta_data_infos[self.filename_map[pure_name]])
            if 'crop' in img_name:
                lw, lh, rw, rh = self.decode_filename(img_name)
                bbox = data_info['ann']['bboxes']
                label = data_info['ann']['labels']
                valid_mask = ((bbox[:, 0] < rw) & (bbox[:, 2] >= lw) &
                              (bbox[:, 1] < rh) & (bbox[:, 3] >= lh))
                bbox, label = bbox[valid_mask], label[valid_mask]
                bbox[:, ::2] = np.clip(bbox[:, ::2], lw, rw - 1) - lw
                bbox[:, 1::2] = np.clip(bbox[:, 1::2], lh, rh - 1) - lh
                data_info['height'], data_info['width'] = rh - lh, rw - lw
                data_info['filename'] = img_name
                data_info['ann']['labels'] = label
                data_info['ann']['bboxes'] = bbox
                if self.filter_empty_gt and len(bbox) == 0:
                    continue

            data_infos.append(data_info)

        return data_infos

    def merge_results(self, results):
        """Merge patch-leval test result to image-level results."""
        concat_results = [[] for _ in range(len(self.filename_map))]
        filename_list = ['' for _ in range(len(self.filename_map))]
        for i in range(len(results)):
            filename = self.data_infos[i]['filename']
            pure_filename = filename.split('@')[-1]
            if 'crop' in filename:
                lw, lh, rw, rh = self.decode_filename(filename)
            else:
                lw = lh = 0
            idx = self.filename_map[pure_filename]
            for (j, cls_res) in enumerate(results[i]):
                cls_res[:, 0] += lw
                cls_res[:, 2] += lw
                cls_res[:, 1::2] += lh
                results[i][j] = cls_res[cls_res[:, -1] >= self.score_threshold]

            concat_results[idx].append(results[i])
            filename_list[idx] = pure_filename

        new_results = []
        for result_list in concat_results:
            # if len(result_list) == 0: continue
            cls_bboxes, class_num = [], len(result_list[0])
            for cls in range(class_num):
                cls_bboxes.append(
                    np.concatenate([res[cls] for res in result_list]))
            nms_bboxes = torch.Tensor(np.concatenate(cls_bboxes)).cuda()
            nms_class = [[i] * len(cls_bboxes[i]) for i in range(class_num)]
            nms_class = torch.cat(
                [nms_bboxes.new_tensor(nms_cls) for nms_cls in nms_class])
            nms_scores = nms_bboxes.new_zeros(nms_bboxes.shape[0],
                                              class_num + 1)
            nms_scores.scatter_(1,
                                nms_class.long()[..., None], nms_bboxes[:, -1,
                                                             None])
            bboxes, labels = multiclass_nms(nms_bboxes[:, :4], nms_scores, 0,
                                            self.nms_cfg)
            if bboxes.shape[0] == 0:
                bboxes = bboxes.new_empty(size=(0, 5))
            new_results.append([
                bboxes[labels == cls].cpu().numpy() for cls in range(class_num)
            ])
        return new_results, filename_list

    def format_results(self, results, out_path='submit.json',
                       neg_num=60, **kwargs):
        # mmcv.dump(results, 'for_debug.pkl')
        results, filename_list = self.merge_results(results)
        max_scores = [np.concatenate(res) for res in results]
        max_scores = [np.max(score[:, -1]) if len(score) else 0.0
                      for score in max_scores]
        score_thr = sorted(max_scores)[neg_num]
        valid_inds = (np.array(max_scores) >= score_thr).nonzero()[0]
        submit_json = []
        for i in valid_inds:
            result, single_json = results[i], []
            for (c, cls_result) in enumerate(result):
                single_json.extend([
                    dict(
                        name=filename_list[i],
                        category=c + 1,
                        bbox=item[:4].tolist(),
                        score=item[4])
                    for item in cls_result
                ])
            submit_json.extend(single_json)
        logger = get_root_logger()
        logger.info(f'json file saved at {out_path}.')
        mmcv.dump(submit_json, out_path)

    def visualize_gt(self, dump_path):
        """visualize data"""
        for data_info in self.data_infos:
            filepath = self.img_prefix + data_info['filename']
            img = mmcv.imread(filepath)
            bboxes = data_info['ann']['bboxes']
            labels = data_info['ann']['labels']
            mmcv.imshow_det_bboxes(
                img, bboxes, labels, class_names=self.CLASSES, show=False)
            mmcv.imwrite(img, f'{dump_path}/{data_info["filename"]}')

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=[0.1, 0.3, 0.5],
                 scale_ranges=None):
        """score=0.8*mAP_score + 0.2*accuracy_score

        mAP score only cares iou_thr=(0.1, 0.3, 0.5),
        Accuracy is the binary accuracy, i.e. w/ or w/o defect.
        """
        new_results, _ = self.merge_results(results)
        eval_results = OrderedDict()
        annotations = [info['ann'] for info in self.meta_data_infos]
        cls_results = [sum([len(k) for k in res]) > 0 for res in new_results]
        cls_targets = [len(ann['labels']) > 0 for ann in annotations]
        cls_results, cls_targets = np.array(cls_results), np.array(cls_targets)
        assert len(cls_results) == len(cls_targets)
        accu = 100 * sum(cls_results == cls_targets) / len(cls_results)
        eval_results['final_score'] = 0.0
        eval_results['accu'] = accu
        eval_results['mAP'] = 0.0
        for iou in iou_thr:
            mean_ap, _ = eval_map(
                new_results,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=iou,
                dataset=self.CLASSES,
                logger=logger)
            eval_results[f'mAP_{iou}'] = mean_ap
            eval_results['mAP'] += mean_ap / len(iou_thr)
        eval_results['final_score'] = (80 * eval_results['mAP'] +
                                       0.2 * eval_results['accu'])
        return eval_results
