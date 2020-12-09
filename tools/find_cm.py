import itertools
import os
import re

import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
from mmcv import color_val


def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      thickness=1,
                      font_scale=0.5,
                      show=True,
                      win_name='',
                      wait_time=0,
                      out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.

    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)
    img = np.ascontiguousarray(img)
    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(
            img, left_top, right_bottom, bbox_color, thickness=thickness)
        label_text = class_names[
            label] if class_names is not None else f'cls {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2)

    if show:
        cv2.imshow(img, win_name, wait_time)
    if out_file is not None:
        cv2.imwrite(img, out_file)
    return img


workdir_path = 'laryngoscopy_output/faster_rcnn_r50_fpn_2x_keep-normal_laryngoscopy_augment'
CLASSES = ('Carcinoma', 'PreCancer', 'Cyst', 'Pol&Nod', 'Normal')


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=15, rotation=45)
    plt.yticks(tick_marks, classes, fontsize=15)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)


confusion_matrix = np.zeros(shape=(5, 5))
mistake_filename = []
pred_list = []

for fold_idx in range(1, 6):
    fold_dir = f'{workdir_path}/fold{fold_idx}/'
    pred = joblib.load(f'{fold_dir}/roc_test_result.pkl')
    pred_list.extend(pred)

correct, wrong = 0, 0

# for pred in pred_list:
#     print(f'correct={correct}, wrong={wrong}')
#     if pred['pred_label'] == pred['gt_label']:
#         correct += 1
#     else:
#         wrong += 1
#     img = cv2.imread(f'data/laryngoscopy/image/{pred["filename"]}')
#     if len(pred['gt_bbox']):
#         imshow_det_bboxes(
#             img,
#             pred['gt_bbox'],
#             np.array(pred['gt_label'])[None],
#             bbox_color='green',
#             text_color='green',
#             thickness=2,
#             font_scale=1.5,
#             class_names=CLASSES,
#             show=False,
#         )
#     if len(pred['pred_bbox']) and pred['pred_bbox'][4] > 0.3:
#         imshow_det_bboxes(
#             img,
#             pred['pred_bbox'][None][:, :5],
#             np.array(pred['pred_label'])[None],
#             bbox_color='blue',
#             text_color='blue',
#             thickness=2,
#             font_scale=1.5,
#             class_names=CLASSES,
#             show=False,
#         )
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # cv2.imwrite(f'./det_vis/{"correct" if pred["pred_label"] == pred["gt_label"] else "wrong"}/{pred["filename"]}', img)
#     cv2.imwrite(f'./det_vis/cascade_rcnn_r50_fro0_fpn_2x_laryngoscopy_augment/{pred["filename"]}', img)

for pred in pred_list:
    gt_label = pred['gt_label']
    pred_label = pred['pred_label']
    confusion_matrix[gt_label][pred_label] += 1

cm = confusion_matrix
total_samples = sum(sum(cm))
eval_res = {'cm': cm, 'precision': [], 'recall': [], 'specific': []}

for cls in range(5):
    TP = cm[cls, cls]
    FN = cm[cls].sum() - TP
    FP = cm[:, cls].sum() - TP
    TN = total_samples - FN - FP - TP
    eval_res['precision'].append(TP / (1e-6 + TP + FP))
    eval_res['recall'].append(TP / (1e-6 + TP + FN))
    eval_res['specific'].append(TN / (1e-6 + TN + FP))

eval_res['precision'].append(round(np.array(eval_res['precision']).mean(), 4))
eval_res['recall'].append(round(np.array(eval_res['recall']).mean(), 4))
eval_res['specific'].append(round(np.array(eval_res['specific']).mean(), 4))

star_num = 67
print('')
print('*' * star_num)
print("| %5s | %-20s | %10s | %7s | %9s *" % ('Top-k', 'Category', 'Precision', 'Recall', 'Specific'))
print('*' * star_num)

for idx in range(6):
    print("| %5s | %-20s | %10s | %7s | %9s |" % (
        f'top-{1}', (CLASSES + ('Average',))[idx],
        "%.2f %%" % (eval_res['precision'][idx] * 100),
        "%.2f %%" % (eval_res['recall'][idx] * 100),
        "%.2f %%" % (eval_res['specific'][idx] * 100),
    ))

print('*' * star_num)

plt.figure(figsize=(10, 10))
plot_confusion_matrix(cm.astype(np.int), CLASSES)
plt.show()

for class_id, class_name in enumerate(CLASSES):
    score_list = [pred['score'][class_id] for pred in pred_list]
    score_list = sorted(score_list)
    roc_cor = []
    for thr in score_list:
        TP, FP, FN, TN = 0, 0, 0, 0
        for pred in pred_list:
            gt_label = pred['gt_label']
            gt_score = pred['score'][class_id]
            if gt_label == class_id:
                if gt_score >= thr:
                    TP += 1
                else:
                    FN += 1
            else:
                if gt_score >= thr:
                    FP += 1
                else:
                    TN += 1
        roc_cor.append((FP / (FP + TN), TP / (TP + FN)))
    area = 0
    for _ in range(1, len(roc_cor)):
        area += (roc_cor[_ - 1][0] - roc_cor[_][0]
                 ) * roc_cor[_ - 1][1]
    roc_cor = np.array(roc_cor)
    plt.xlabel('FP ratio')
    plt.ylabel('TP ratio')
    plt.title(f'{class_name}, AUC={round(area, 4)}')
    plt.plot(roc_cor[:, 0], roc_cor[:, 1])
    plt.show()



# for config in os.listdir(workdir_path):
#     test_fold_list, cur_test_fold = [], None
#     confusion_matrix = np.zeros(shape=(25))
#     for file_name in os.listdir(workdir_path + config):
#         if not file_name.endswith('.log'):
#             continue
#         print(file_name)
#         with open(workdir_path + config + '/' + file_name, 'r') as f:
#             content = f.readlines()
#         best_epoch = content[-1].strip()
#         # best_epoch = re.findall(r"(AP-best: \(\d+(\.\d+)?, \d+)", best_epoch)
#         best_epoch = re.findall(r"(Ave-best: \(\d+(\.\d+)?, \d+)", best_epoch)
#         best_epoch = re.findall(r'\d+', best_epoch[0][0])[-1]
#         if len(best_epoch) == 0:
#             continue
#         for line in content:
#             if 'test_fold' in line:
#                 cur_test_fold = line
#         # if cur_test_fold in test_fold_list:
#         #     continue
#         test_fold_list.append(cur_test_fold)
#         best_epoch = int(best_epoch)
#
#
#
#
#
#         # best_epoch = 12
#
#
#
#
#         for i, line in enumerate(content):
#             line = line.strip()
#             re_pattern = f'Epoch\(val\) \[{best_epoch}\]'
#             if re.findall(re_pattern, line):
#                 matrix = []
#                 for k in range(7, 1, -1):
#                     line = content[i - k].strip()
#                     line = re.findall(r'\[.*\]', line)[0]
#                     line = re.findall(r'\d+', line)
#                     matrix.extend([int(x) for x in line])
#
#                 confusion_matrix += np.array(matrix)
#                 print(best_epoch, np.array(matrix).reshape((6, 6)))
#                 # print(matrix)
#                 # print(np.array(matrix).reshape((6,6)))
#     print(confusion_matrix.reshape((6, 6)))
#     # break
#     # matrix = [ matrix[_*6: _*6+6] for _ in range(6) ]
#     # print(matrix)
#     # print(confusion_matrix.reshape((6, 6)))
#     print('')
#     eval_results = []
#     cm = confusion_matrix.reshape((6, 6))
#     # temp = [2023, 17, 0, 1, 2, 19, 1228, 15, 2, 0, 1, 9, 541, 3, 2, 8, 13, 1, 617, 26, 16, 0, 40, 22, 628]
#     # temp = [1032, 10, 17, 1, 8, 8, 124, 541, 378, 24, 3, 6, 13, 35, 1036, 14, 53, 29, 24, 14, 60, 783, 114, 39, 25, 0, 199, 29, 652, 202]
#     cnt = -1
#     # cm *= 0
#     for i in range(5):
#         for j in range(6):
#             cnt += 1
#             # print(i, j, cnt)
#             # cm[i][j] = temp[cnt]
#     # cm = cm.astype(np.int)
#     # print(cm)
#     eval_res = {'k': 1, 'cm': cm,
#                 'precision': [], 'recall': [],
#                 'specific': []}
#     total_samples = sum(sum(cm))
#     for cls in range(6):
#         TP = cm[cls, cls]
#         FN = cm[cls].sum() - TP
#         FP = cm[:, cls].sum() - TP
#         TN = total_samples - FN - FP - TP
#         eval_res['precision'].append(TP / (1e-6 + TP + FP))
#         eval_res['recall'].append(TP / (1e-6 + TP + FN))
#         eval_res['specific'].append(TN / (1e-6 + TN + FP))
#
#     eval_res['precision'].append(round(np.array(eval_res['precision']).mean(), 4))
#     eval_res['recall'].append(round(np.array(eval_res['recall']).mean(), 4))
#     eval_res['specific'].append(round(np.array(eval_res['specific']).mean(), 4))
#     eval_results.append(eval_res)
#     star_num = 67
#     print('')
#     print('*' * star_num)
#     print("| %5s | %-20s | %10s | %7s | %9s *" % ('Top-k', 'Category', 'Precision', 'Recall', 'Specific'))
#     print('*' * star_num)
#     # CLASSES = ('Normal', 'Nodule', 'Polyps', 'Leukoplakia', 'Malignant', 'Average')
#     CLASSES = ('Laryngeal Carcinoma', 'Cord Cyst', 'Nodules', 'Polyps', 'Leukoplakia', 'Normal', 'Average')
#     for idx in range(7):
#         # print("| %5s | %-20s | %10s | %7s |" % (
#         #     f'top-{topks[i]}', CLASSES[idx],
#         #     "%.2f %%" % (eval_res['precision'][idx] * 100),
#         #     "%.2f %%" % (eval_res['recall'][idx] * 100),
#         # ))
#         print("| %5s | %-20s | %10s | %7s | %9s |" % (
#             f'top-{1}', CLASSES[idx],
#             "%.2f %%" % (eval_res['precision'][idx] * 100),
#             "%.2f %%" % (eval_res['recall'][idx] * 100),
#             "%.2f %%" % (eval_res['specific'][idx] * 100),
#         ))
#
#     print('*' * star_num)
#
#     plt.figure(figsize=(10, 10))
#     plot_confusion_matrix(cm[:6, :6].astype(np.int), CLASSES[:6])
#     plt.show()
#     break
#
# # 2020 - 10 - 13
# # 22: 01:10, 614 - mmdet - INFO - Epoch(val)[12][129]
# # AP - top1: 0.4628, AR - top1: 0.2858, AP - best: (0.4655, 10), AR - best: (0.3283, 8)
#
# #     # print(content)
# #     break
# # break
