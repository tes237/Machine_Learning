import cv2 as cv
import numpy as np
import torch
from torchvision import datasets, transforms
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')

        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows, cols)
        return images

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num = int.from_bytes(f.read(4), 'big')

        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels


# =========================
# 2️⃣ Deskew
# =========================
SZ = 28
affine_flags = cv.WARP_INVERSE_MAP | cv.INTER_LINEAR

def deskew(img):
    m = cv.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SZ * skew],
                    [0, 1, 0]])
    img = cv.warpAffine(img, M, (SZ, SZ), flags=affine_flags)
    return img

# 3) HOG feature extraction function
bin_n = 16

def hog(img):
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)

    bins = np.int32(bin_n * ang / (2 * np.pi))
    bins = np.clip(bins, 0, bin_n - 1)

    bin_cells = bins[:14,:14], bins[14:,:14], bins[:14,14:], bins[14:,14:]
    mag_cells = mag[:14,:14], mag[14:,:14], mag[:14,14:], mag[14:,14:]

    hists = [np.bincount(b.ravel(), m.ravel(), bin_n)
            for b, m in zip(bin_cells, mag_cells)]

    hist = np.hstack(hists)
    return hist

def svm_main():

    # Define a transform to convert the images to tensors and normalize
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

    # Load the training dataset
    train_dataset = datasets.MNIST(root='../Data/mnist_data', train=True, download=True, transform=transform)

    # Load the test dataset
    test_dataset = datasets.MNIST(root='../Data/mnist_data', train=False, download=True, transform=transform)

    train_images = load_mnist_images("../Data/mnist_data/MNIST/raw/train-images-idx3-ubyte")
    train_labels = load_mnist_labels("../Data/mnist_data/MNIST/raw/train-labels-idx1-ubyte")

    test_images = load_mnist_images("../Data/mnist_data/MNIST/raw/t10k-images-idx3-ubyte")
    test_labels = load_mnist_labels("../Data/mnist_data/MNIST/raw/t10k-labels-idx1-ubyte")

    print("train_images structure : " ,  train_images.shape)  # (60000, 28, 28)
    print("train_labels structure : " ,  train_labels.shape)  # (60000,)

    print("test_images structure : " ,  test_images.shape)  
    print("test_labels structure : " ,  test_labels.shape)  

    #Diminish Data size
    #train_images  = train_images [:10000]
    #train_labels = train_labels[:10000]

    #label reshape
    train_labels = train_labels.reshape(-1, 1)

    train_data = []
    for img in train_images:
        img = deskew(img)
        feature = hog(img)
        train_data.append(feature)

    test_data = []
    for img in test_images:
        img = deskew(img)
        feature = hog(img)
        test_data.append(feature)

    # type conversion
    train_data = np.float32(train_data)
    train_labels = np.int32(train_labels)
    test_data = np.float32(test_data)
    test_labels = np.int32(test_labels)

    # C/Gamma range for hyper parameter tuning
    C = [1, 2.5, 5, 10]
    Gamma = [0.0005, 0.001, 0.002]

    print(train_images.shape)
    print(test_images.shape)
    print(np.bincount(test_labels))

    print("train_features:", train_data.shape, train_data.dtype)
    print("train_labels:", train_labels.shape, train_labels.dtype)
    print("unique labels:", np.unique(train_labels))

    # Regularization with Scaler
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data  = scaler.transform(test_data)

    # 6) SVM Training
    for tmpC in C:
        for tmpGamma in Gamma:
            svm = cv.ml.SVM_create()
            svm.setType(cv.ml.SVM_C_SVC)
            svm.setKernel(cv.ml.SVM_RBF)
            #svm.setKernel(cv.ml.SVM_LINEAR)
            
            svm.setC(tmpC)
            svm.setGamma(tmpGamma)
            
            print("before train : ", datetime.now())
            svm.train(train_data, cv.ml.ROW_SAMPLE, train_labels)
            print("after train : ", datetime.now())

            # 7) predict & calculate accuracy calculation
            #print("before predict : ", datetime.now())
            ret, result = svm.predict(test_data)
            #print("after predict : ", datetime.now())

            print(np.bincount(result.astype(int).flatten()))

            matches = result.flatten() == test_labels
            correct = np.count_nonzero(matches)

            print("C = {:.2f}".format(tmpC), ", Gamma = {:.4f}".format(tmpGamma), ", Accuracy: {:.2f}%".format(correct * 100.0 / len(test_labels)))


if __name__ == "__main__":
    svm_main()

