import numpy as np
import torch
from torchvision import datasets, transforms
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import math
import cv2 
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import psutil
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

LABELS = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass","Nodule",
    "Pneumonia","Pneumothorax","Consolidation","Edema","Emphysema","Fibrosis",
    "Pleural_Thickening","Hernia"
]

def monitor_system_memory():
    """Prints the total, available, used, and percentage of system RAM."""
    mem = psutil.virtual_memory()
    print(f"Total RAM: {mem.total / (1024**3):.2f} GB")
    print(f"Available RAM: {mem.available / (1024**3):.2f} GB")
    print(f"Used RAM: {mem.used / (1024**3):.2f} GB")
    print(f"Usage Percentage: {mem.percent}%")

    swap = psutil.swap_memory()
    print(f"Total Swap: {swap.total / (1024**3):.2f} GB")
    print(f"Used Swap: {swap.used / (1024**3):.2f} GB")
    print(f"Free Swap: {swap.free / (1024**3):.2f} GB")
    print(f"Swap Usage Percent: {swap.percent}%")
    print(f"Swap In (bytes/sec): {swap.sin}")
    print(f"Swap Out (bytes/sec): {swap.sout}")

def get_diagnosis(diagnosis_label):
    match diagnosis_label:
        case "Atelectasis":
            return 0
        case "Cardiomegaly":
            return 1
        case "Effusion":
            return 2
        case "Infiltration":
            return 3
        case "Mass":
            return 4
        case "Nodule":
            return 5
        case "Pneumonia":
            return 6
        case "Pneumothorax":
            return 7
        case "Consolidation":
            return 8
        case "Edema":
            return 9
        case "Emphysema":
            return 10
        case "Fibrosis":
            return 11
        case "Pleural_Thickening":
            return 12
        case "Hernia":
            return 13
        case _: # No Finding
            return 14
        
def svm_main():

    # for windows
    #data_dir = 'C:\\src\\python\\uofc\\ml_project\\Data\\archive\\images_001\\images\\'
    #data_csv = 'C:\\src\\python\\uofc\\ml_project\\Data\\archive\\image001_Data_Entry_2017.csv'

    # for linux
    data_dir = '/home/developer/src/python/UofC/ml_project/Data/archive/image_test/images/'
    data_csv = '/home/developer/src/python/UofC/ml_project/Data/archive/Data_Entry_2017.csv'

    #data_dir = '/home/developer/src/python/UofC/ml_project/Data/archive/image_mini_test/images/'
    #data_csv = '/home/developer/src/python/UofC/ml_project/Data/archive/minitest_Data_Entry_2017.csv'

    all_files = []
    #Y = np.zeros((4999, 14))
    Y_rows = []
    Y = np.zeros((1, 14))

    with open(data_csv, mode='r') as file_feat:

        loop_index = 0
        line_index = 0
        for line in file_feat:
            if(line_index == 0):
                line_index += 1
                continue

            parts = line.split(",")
            image_file_name = parts[0]
            tmp_part = parts[1]

            row = np.zeros(14)
            if (tmp_part.find("|") >= 0):
                diagnosis_arr = tmp_part.split("|")

                tmp_diag_index = 0
                for tmp_diagnosis in diagnosis_arr:
                    diagnosis = get_diagnosis(tmp_diagnosis)
                    if(diagnosis != 14):
                        row[diagnosis] = 1

            else:
                diagnosis = get_diagnosis(tmp_part)
                if(diagnosis != 14):
                    row[diagnosis] = 1

            Y_rows.append(row)

            all_files.append(data_dir + image_file_name)
            loop_index += 1

        Y = np.array(Y_rows)
        
        number_of_samples = len(all_files)

        train_length = math.floor(len(all_files) * 0.8)
        #eval_length = math.floor(len(all_files) * 0.1)
        test_length = math.floor(len(all_files) * 0.2)

        train_target_files = []
        train_target_Y = []
        train_len = 0

        test_target_files = []
        test_target_Y = []
        test_len = 0

        train_target_files, test_target_files, train_target_Y, test_target_Y = train_test_split(
            all_files, Y, test_size=0.2, random_state=42
        )


        train_target_Y = np.array(train_target_Y)
        train_len = len(train_target_files)
        test_target_Y = np.array(test_target_Y)
        test_len = len(test_target_files)

    QUARTER_IMAGE_SIZE = 256
    IMAGE_SIZE = 224
    train_data = []
    for image_path in train_target_files:

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_quarter_resized = cv2.resize(img, (QUARTER_IMAGE_SIZE, QUARTER_IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        crop_size = IMAGE_SIZE
        start = (QUARTER_IMAGE_SIZE - crop_size) // 2
        img_resized = img_quarter_resized[start:start+crop_size, start:start+crop_size]
        
        feature = hog(img_resized,
                    pixels_per_cell=(4,4),
                    cells_per_block=(2,2),
                    block_norm='L2-Hys',
                    feature_vector=True)
        
        #feature = hog(img)

        train_data.append(feature)

    test_data = []
    for image_path in test_target_files:

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_quarter_resized = cv2.resize(img, (QUARTER_IMAGE_SIZE, QUARTER_IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        crop_size = IMAGE_SIZE
        start = (QUARTER_IMAGE_SIZE - crop_size) // 2
        img_resized = img_quarter_resized[start:start+crop_size, start:start+crop_size]

        feature = hog(img_resized,
                    pixels_per_cell=(4,4),
                    cells_per_block=(2,2),
                    block_norm='L2-Hys',
                    feature_vector=True)

        #feature = hog(img)
        test_data.append(feature)

    # type conversion
    train_data = np.float32(train_data)
    train_labels = np.int32(train_target_Y)
    
    test_data = np.float32(test_data)
    test_labels = np.int32(test_target_Y)

    # C/Gamma range for hyper parameter tuning
    C = [1, 2.5, 5, 10]
    #Gamma = [0.0005, 0.001, 0.002]
    
    print("train_features:", train_data.shape, train_data.dtype)
    print("train_labels:", train_labels.shape, train_labels.dtype)
    print("unique labels:", np.unique(train_labels))

    # 6) SVM Training
    #for tmpC in C:
    tmpC = 1

    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=500,
            svd_solver='randomized',
            random_state=42
        ),
        OneVsRestClassifier(
            LinearSVC(C = tmpC, class_weight='balanced', max_iter=20000)
        )
    )

    monitor_system_memory()
    print("before train : ", datetime.now())
    model.fit(train_data, train_labels)
    print("after train : ", datetime.now())
    monitor_system_memory()

    # 7) predict & calculate accuracy calculation
    #result = model.predict(test_data)

    scores = model.decision_function(test_data)

    print("decision_function scores min : ", scores.min())
    print("decision_function scores max : ", scores.max())
    print("decision_function scores mean : ", scores.mean())

    #threshold = -0.2
    #result = (scores > threshold).astype(int)

    thresholds = [
    -0.3,  # 0
    -0.2,  # 1
    -0.3,  # 2
    -0.2,  # 3
    -0.1,  # 4
    -0.1,  # 5
    0.0,  # 6
    -0.1,  # 7
    -0.1,  # 8
    -0.1,  # 9
    -0.1,  # 10
    -0.1,  # 11
    0.2,  # 12 (rare → stricter)
    -0.3   # 13
    ]

    result = (scores > thresholds).astype(int)

    print(classification_report(test_labels, result))
    
    #ROC AUC score

    test_auc_per_class = roc_auc_score(test_labels, scores, average=None)
    test_mean_auc = roc_auc_score(test_labels, scores, average="macro")

    print("\nTest Mean AUROC:", test_mean_auc)
    for name, auc_val in zip(LABELS, test_auc_per_class):
        print(f"TEST {name:18s}: {auc_val:.4f}")

    print(train_data.shape)
    print(train_labels.shape)
    print(np.sum(train_labels, axis=0))



    n_classes = len(LABELS)

    # Per-class ROC
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)

    # Std band across classes
    tpr_interp_all = []
    for i in range(n_classes):
        tpr_interp_all.append(np.interp(all_fpr, fpr[i], tpr[i]))

    tpr_interp_all = np.array(tpr_interp_all)
    std_tpr = np.std(tpr_interp_all, axis=0)

    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: per-class ROC curves
    for i in range(n_classes):
        axes[0].plot(fpr[i], tpr[i], lw=1.5, label=f"{LABELS[i]} ({roc_auc[i]:.3f})")

    axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label="Random")
    axes[0].set_title("Per-class ROC Curves")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1.02])
    axes[0].legend(loc="lower right", fontsize=8)

    # Right: macro-average ROC
    axes[1].plot(all_fpr, mean_tpr, lw=2, label=f"Macro-avg ROC (AUC = {macro_auc:.4f})")
    axes[1].fill_between(all_fpr, tpr_lower, tpr_upper, alpha=0.2, label="± 1 std across classes")
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1, label="Random")
    axes[1].set_title("Macro-Average ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1.02])
    axes[1].legend(loc="lower right", fontsize=9)

    plt.suptitle("ROC / AUC Analysis — NIH ChestXray14", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("SVC_roc_auc_curves.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    svm_main()

