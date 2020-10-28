import itertools
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# folder = 'configs/laryngoscopy/'
#
# for config_name in os.listdir(folder):
#     print(f"python tools/train.py {folder}{config_name}")

workdir_path = './work_dirs/'
# workdir_path = './laryngoscopy_benchmark/'


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


for config in os.listdir(workdir_path):
    if 'laryngoscopy' not in config:
        continue
    print(config)
    test_fold_list, cur_test_fold = [], None
    confusion_matrix = np.zeros(shape=(36))
    for file_name in os.listdir(workdir_path + config):
        if not file_name.endswith('.log'):
            continue
        print(file_name)
        with open(workdir_path + config + '/' + file_name, 'r') as f:
            content = f.readlines()
        best_epoch = content[-1].strip()
        best_epoch = re.findall(r"(AP-best: \(\d+(\.\d+)?, \d+)",
                                best_epoch)
        best_epoch = re.findall(r'\d+', best_epoch[0][0])[-1]
        if len(best_epoch) == 0:
            continue
        for line in content:
            if 'test_fold' in line:
                cur_test_fold = line
        # if cur_test_fold in test_fold_list:
        #     continue
        test_fold_list.append(cur_test_fold)
        best_epoch = int(best_epoch)





        best_epoch = 10




        for i, line in enumerate(content):
            line = line.strip()
            re_pattern = f'Epoch\(val\) \[{best_epoch}\]'
            if re.findall(re_pattern, line):
                matrix = []
                for k in range(6, 0, -1):
                    line = content[i - k].strip()
                    line = re.findall(r'\[.*\]', line)[0]
                    line = re.findall(r'\d+', line)
                    matrix.extend([int(x) for x in line])

                confusion_matrix += np.array(matrix)
                print(best_epoch, np.array(matrix).reshape((6, 6)))
                # print(matrix)
                # print(np.array(matrix).reshape((6,6)))
    print(confusion_matrix.reshape((6, 6)))
    # break
    # matrix = [ matrix[_*6: _*6+6] for _ in range(6) ]
    # print(matrix)
    # print(confusion_matrix.reshape((6, 6)))
    print('')
    eval_results = []
    cm = confusion_matrix.reshape((6, 6))
    # temp = [2023, 17, 0, 1, 2, 19, 1228, 15, 2, 0, 1, 9, 541, 3, 2, 8, 13, 1, 617, 26, 16, 0, 40, 22, 628]
    # temp = [1032, 10, 17, 1, 8, 8, 124, 541, 378, 24, 3, 6, 13, 35, 1036, 14, 53, 29, 24, 14, 60, 783, 114, 39, 25, 0, 199, 29, 652, 202]
    cnt = -1
    # cm *= 0
    for i in range(5):
        for j in range(6):
            cnt += 1
            # print(i, j, cnt)
            # cm[i][j] = temp[cnt]
    # cm = cm.astype(np.int)
    # print(cm)
    eval_res = {'k': 1, 'cm': cm,
                'precision': [], 'recall': [],
                'specific': []}
    total_samples = sum(sum(cm))
    for cls in range(6):
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
    eval_results.append(eval_res)
    star_num = 67
    print('')
    print('*' * star_num)
    print("| %5s | %-20s | %10s | %7s | %9s *" % ('Top-k', 'Category', 'Precision', 'Recall', 'Specific'))
    print('*' * star_num)
    # CLASSES = ('Normal', 'Nodule', 'Polyps', 'Leukoplakia', 'Malignant', 'Average')
    CLASSES = ('Laryngeal Carcinoma', 'Cord Cyst', 'Nodules', 'Polyps', 'Leukoplakia', 'Normal', 'Average')
    for idx in range(7):
        # print("| %5s | %-20s | %10s | %7s |" % (
        #     f'top-{topks[i]}', CLASSES[idx],
        #     "%.2f %%" % (eval_res['precision'][idx] * 100),
        #     "%.2f %%" % (eval_res['recall'][idx] * 100),
        # ))
        print("| %5s | %-20s | %10s | %7s | %9s |" % (
            f'top-{1}', CLASSES[idx],
            "%.2f %%" % (eval_res['precision'][idx] * 100),
            "%.2f %%" % (eval_res['recall'][idx] * 100),
            "%.2f %%" % (eval_res['specific'][idx] * 100),
        ))

    print('*' * star_num)

    # plt.figure(figsize=(10, 10))
    # plot_confusion_matrix(cm[:5, :5].astype(np.int), CLASSES[:5])
    # plt.show()

# 2020 - 10 - 13
# 22: 01:10, 614 - mmdet - INFO - Epoch(val)[12][129]
# AP - top1: 0.4628, AR - top1: 0.2858, AP - best: (0.4655, 10), AR - best: (0.3283, 8)

#     # print(content)
#     break
# break
