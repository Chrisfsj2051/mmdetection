import itertools
import os
import re

import joblib
import numpy as np
import matplotlib.pyplot as plt

workdir_path = './work_dirs/'

for config in os.listdir(workdir_path):
    if 'laryngoscopy' not in config:
        continue
    print(config)
    total_pred, total_gt = [], []
    for file_name in os.listdir(workdir_path + config):
        if not file_name.endswith('.pkl'):
            continue
        loaded = joblib.load(workdir_path + config + '/' + file_name)
        pred_res, gt_label = loaded['test_result'], loaded['gt_labels']
        total_pred.extend(pred_res)
        total_gt.extend(gt_label)
    CLASSES=('Laryngeal Carcinoma','Cord Cyst','Nodules','Polyps','Leukoplakia')
    for c in range(5):
        pred_as_gt = []
        roc_cor = []
        for i in range(len(total_gt)):
            gt, p = total_gt[i], total_pred[i]
            pred_as_gt.append(p[0] if int(p[1]) == c else np.random.uniform(0, 0.3))
        pred_as_gt.append(1.0)
        pred_as_gt.append(0.0)
        for thr in np.unique(np.array(sorted(pred_as_gt))):
            TP, FP, FN, TN = 0, 0, 0, 0
            for i in range(len(total_gt)):
                gt, p = total_gt[i], pred_as_gt[i]
                if gt == c:
                    if p >= thr:
                        TP += 1
                    else:
                        FN += 1
                else:
                    if p >= thr:
                        FP += 1
                    else:
                        TN += 1
            roc_cor.append((FP / (FP + TN), TP / (TP + FN)))

        roc_cor = np.array(roc_cor)
        plt.xlabel('FP ratio')
        plt.ylabel('TP ratio')
        plt.title(CLASSES[c-1])
        plt.plot(roc_cor[:, 0], roc_cor[:, 1])
        plt.show()


            # gt, p = total_gt[i], total_pred[i]
            # if gt == c and p[0] == c:
            #     TP += 1
            #     predict.append(p[1])
            # elif gt == c and p[0] != c:
            #     FN += 1

