import cv2 as cv
import numpy as np

import torch
from torchvision import datasets, transforms


import numpy as np

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


def svm_main():
    '''
    # 학습 데이터 (4개 포인트)
    labels = np.array([1, -1, -1, -1], dtype=np.int32)
    trainingData = np.array([[501, 10], [255, 10], [501, 255], [10, 501]], dtype=np.float32)

    # SVM 객체 생성
    svm = cv.ml.SVM_create()

    # SVM 타입과 커널 설정
    svm.setType(cv.ml.SVM_C_SVC)

    # SVM_CUSTOM: int
    # SVM_LINEAR: int
    # SVM_POLY: int
    # SVM_RBF: int
    # SVM_SIGMOID: int
    # SVM_CHI2: int
    # SVM_INTER: int
    svm.setKernel(cv.ml.SVM_LINEAR)  # LINEAR, RBF, POLY 등 선택 가능

    # 학습 종료 조건 설정
    svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

    # SVM 학습
    svm.train(trainingData, cv.ml.ROW_SAMPLE, labels)

    # 학습한 모델로 예측
    sample = np.array([[100, 200], [400, 100]], dtype=np.float32)
    ret, result = svm.predict(sample)

    print("예측 결과:", result.ravel())
    '''

    # Define a transform to convert the images to tensors and normalize
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

    # Load the training dataset
    batch_size = 4
    train_dataset = datasets.MNIST(root='../Data/mnist_data', train=True, download=True, transform=transform)
    #trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Load the test dataset
    test_dataset = datasets.MNIST(root='../Data/mnist_data', train=False, download=True, transform=transform)
    #testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    train_images = load_mnist_images("../Data/mnist_data/MNIST/raw/train-images-idx3-ubyte")
    train_labels = load_mnist_labels("../Data/mnist_data/MNIST/raw/train-labels-idx1-ubyte")

    test_images = load_mnist_images("../Data/mnist_data/MNIST/raw/t10k-images-idx3-ubyte")
    test_labels = load_mnist_labels("../Data/mnist_data/MNIST/raw/t10k-labels-idx1-ubyte")

    print("train_images structure : " ,  train_images.shape)  # (60000, 28, 28)
    print("train_labels structure : " ,  train_labels.shape)  # (60000,)

    print("test_images structure : " ,  test_images.shape)  # (60000, 28, 28)
    print("test_labels structure : " ,  test_labels.shape)  # (60000,)

    train_data = train_images.reshape(-1, 28*28)
    test_data = test_images.reshape(-1, 28*28)

    #정규화
    train_data = train_data / 255.0

    # 반드시 타입 변환
    train_data = np.float32(train_data)
    train_labels = np.int32(train_labels)
    test_data = np.float32(test_data)
    test_labels = np.int32(test_labels)

    #데이터 줄이기
    train_data = train_data[:10000]
    train_labels = train_labels[:10000]

    # 1) 상수 정의
    SZ = 20      # 이미지 크기
    bin_n = 16   # HOG bin 수
    affine_flags = cv.WARP_INVERSE_MAP | cv.INTER_LINEAR

    # 2) 이미지 왜곡 보정 함수
    def deskew(img):
        m = cv.moments(img)
        if abs(m['mu02']) < 1e-2:
            return img.copy()
        skew = m['mu11']/m['mu02']
        M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
        img = cv.warpAffine(img, M, (SZ, SZ), flags=affine_flags)
        return img

    # 3) HOG 특징 추출 함수
    def hog(img):
        gx = cv.Sobel(img, cv.CV_32F, 1, 0)
        gy = cv.Sobel(img, cv.CV_32F, 0, 1)
        mag, ang = cv.cartToPolar(gx, gy)
        bins = np.int32(bin_n * ang / (2 * np.pi))
        bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
        mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)
        return hist

    '''
    # 4) MNIST 데이터 로딩 (예: OpenCV 샘플)
    digits = cv.imread('digits.png', 0)
    # digits를 5000개의 20x20 이미지로 쪼갬
    cells = [np.hsplit(row, 100) for row in np.vsplit(digits, 50)]
    train_cells = [c[:50] for c in cells]
    test_cells  = [c[50:] for c in cells]

    # 5) 특징 & 레이블 구성
    train_data = []
    train_labels = []
    for i, row in enumerate(train_cells):
        for img in row:
            img = deskew(img)
            train_data.append(hog(img))
            train_labels.append(i)
    train_data = np.float32(train_data)
    train_labels = np.int32(train_labels)

    test_data = []
    test_labels = []
    for i, row in enumerate(test_cells):
        for img in row:
            img = deskew(img)
            test_data.append(hog(img))
            test_labels.append(i)
    test_data = np.float32(test_data)
    test_labels = np.int32(test_labels)
    '''

    # 6) SVM 학습
    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_C_SVC)
    #svm.setKernel(cv.ml.SVM_RBF)
    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.setC(12.5)
    #svm.setGamma(0.50625)
    svm.setGamma(0.0013)

    #for batch_idx, (data, target) in enumerate(trainloader):
        # data and target are PyTorch tensors on the CPU by default
        # You may want to move them to a GPU (e.g., data.to(device))
        # ... model training steps ...
        #print("batch index : ", batch_idx)
        #svm.train(data, cv.ml.ROW_SAMPLE, target)
        #pass

    svm.train(train_data, cv.ml.ROW_SAMPLE, train_labels)

    # 7) 테스트 & 정확도 계산
    ret, result = svm.predict(test_data)
    matches = result.flatten() == test_labels
    correct = np.count_nonzero(matches)

    print("정확도: {:.2f}%".format(correct * 100.0 / len(test_labels)))


if __name__ == "__main__":
    svm_main()

