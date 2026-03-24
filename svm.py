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
        case "Pleural_Thickening":
            return 11
        case "Hernia":
            return 12
        case _: # No Finding
            return 13

def svm_main():

    # for windows
    #data_dir = 'C:\\src\\python\\uofc\\ml_project\\Data\\archive\\images_001\\images\\'
    #data_csv = 'C:\\src\\python\\uofc\\ml_project\\Data\\archive\\image001_Data_Entry_2017.csv'

    # for linux
    data_dir = '/home/developer/src/python/UofC/ml_project/Data/archive/image_test/images/'
    data_csv = '/home/developer/src/python/UofC/ml_project/Data/archive/image_test2_Data_Entry_2017.csv'

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
                    #Y[loop_index, diagnosis] = 1
                    row[diagnosis] = 1

            else:
                diagnosis = get_diagnosis(tmp_part)
                #Y[loop_index, diagnosis] = 1
                row[diagnosis] = 1

            Y_rows.append(row)

            all_files.append(data_dir + image_file_name)
            loop_index += 1

        Y = np.array(Y_rows)


        #diminish dataset
        #all_files = all_files[0:1000]
        #Y = Y[0:1000]
        
        number_of_samples = len(all_files)

        train_length = math.floor(len(all_files) * 0.8)
        #eval_length = math.floor(len(all_files) * 0.1)
        test_length = math.floor(len(all_files) * 0.2)

        train_target_files = []
        train_target_Y = []
        train_len = 0

        #eval_target_files = []
        #eval_target_Y = []
        #eval_len = 0

        test_target_files = []
        test_target_Y = []
        test_len = 0

        '''
        for index in range(0, train_length):
            train_target_files.append(all_files[index])
            train_target_Y.append(Y[index])

        train_target_Y = np.array(train_target_Y)

        train_len = len(train_target_files)
    
        for index in range(train_length, train_length + eval_length):
            eval_target_files.append(all_files[index])
            eval_target_Y.append(Y[index])
        
        eval_target_Y = np.array(eval_target_Y)

        eval_len = len(eval_target_files)
    
        #train_type == TRAIN_TYPE_TEST:
        for index in range(train_length + eval_length, train_length + eval_length + test_length):
            test_target_files.append(all_files[index])
            test_target_Y.append(Y[index])

        test_target_Y = np.array(test_target_Y)

        test_len = len(test_target_files)
        '''

        train_target_files, test_target_files, train_target_Y, test_target_Y = train_test_split(
            all_files, Y, test_size=0.2, random_state=42
        )


        train_target_Y = np.array(train_target_Y)
        train_len = len(train_target_files)
        test_target_Y = np.array(test_target_Y)
        test_len = len(test_target_files)

    train_data = []
    for image_path in train_target_files:

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        
        feature = hog(img_resized,
                    pixels_per_cell=(16,16),
                    cells_per_block=(2,2),
                    block_norm='L2-Hys',
                    feature_vector=True)
        
        #feature = hog(img)

        train_data.append(feature)

    test_data = []
    for image_path in test_target_files:

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # shape (H, W), dtype uint8 
        img_resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)

        feature = hog(img_resized,
                    pixels_per_cell=(16,16),
                    cells_per_block=(2,2),
                    block_norm='L2-Hys',
                    feature_vector=True)

        #feature = hog(img)
        test_data.append(feature)

    # type conversion
    train_data = np.float32(train_data)
    #train_labels = np.int32(train_labels)
    train_labels = np.int32(train_target_Y)
    
    test_data = np.float32(test_data)
    #test_labels = np.int32(test_labels)
    test_labels = np.int32(test_target_Y)

    # C/Gamma range for hyper parameter tuning
    C = [1, 2.5, 5, 10]
    #Gamma = [0.0005, 0.001, 0.002]
    
    print("train_features:", train_data.shape, train_data.dtype)
    print("train_labels:", train_labels.shape, train_labels.dtype)
    print("unique labels:", np.unique(train_labels))

    # 6) SVM Training
    for tmpC in C:
        #for tmpGamma in Gamma:

            '''
            model = make_pipeline(
                StandardScaler(),
                PCA(n_components=200),
                OneVsRestClassifier(
                    SVC(kernel='rbf', C = tmpC, gamma = tmpGamma, class_weight='balanced')
                )
            )
            '''
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

            #print("after predict : ", datetime.now())

            #print(np.bincount(result.astype(int).flatten()))
            #matches = result.flatten() == test_labels
            #correct = np.count_nonzero(matches)
            #print("C = {:.2f}".format(tmpC), ", Gamma = {:.4f}".format(tmpGamma), ", Accuracy: {:.2f}%".format(correct * 100.0 / len(test_labels)))
            print(classification_report(test_labels, result))

            print(train_data.shape)
            print(train_labels.shape)
            print(np.sum(train_labels, axis=0))


if __name__ == "__main__":
    svm_main()

