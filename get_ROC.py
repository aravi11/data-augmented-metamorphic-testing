import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score
from itertools import cycle
from my_functions import *
import networkx

mr_type = 'additive'
random_state = np.random.RandomState(0)

#get the Gpickle having the list of CFGS extracted from soot for permutation MR

gpickle_File_Path = './gPickles/additive/additive_org_m1_m3_m4.gpickle'
graph_List = networkx.read_gpickle(gpickle_File_Path)

C_param = 1000
lambda_param = 1.2

#get the labels for Permutation MR

label = np.loadtxt("./Labels/additive/final_labels/additive_org_m1_m3_m4.txt", dtype=np.int32)

data, target = balance_data(graph_List, label, random_state=random_state)

#Define parameters for SVM
skf = StratifiedKFold(n_splits=7)
SVM = SVC(C=C_param, kernel='precomputed', probability=True, random_state=random_state)

tprs = []
aucs = []
AUCS = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train_index, test_index in skf.split(data, target):
    #print("Train_index, Test Index = " + str(train_index) + ", "+str(test_index) + ">>>>>>>>>>>" + str(i))
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = target[train_index], target[test_index]
    #kernel_matix_train = random_walk_kernel_1(X_train, X_train, lmb=0.6)
    #kernel_matix_test = random_walk_kernel_1(X_test, X_train, lmb=0.6)
    kernel_matix_train = compute_kernel_matrix(X_train, X_train, lmb=lambda_param, type="RWK")
    kernel_matix_test = compute_kernel_matrix(X_test, X_train, lmb=lambda_param, type="RWK")
    probas_ = SVM.fit(kernel_matix_train, y_train).predict_proba(kernel_matix_test)
    AUCS.append(roc_auc_score(y_test, probas_[:, 1]))
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    print(fpr)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1

print(np.mean(AUCS))

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Line of no discrimination', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for ' + str(mr_type)+' MR, parameters = [C = ' + str(C_param)+', Lambda = ' + str(lambda_param)+ ']')
plt.legend(loc="lower right")
plt.show()

