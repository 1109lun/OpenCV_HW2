import sys
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QWidget, QVBoxLayout, QLabel , QMessageBox, QGraphicsScene, QGraphicsView 
from PyQt5.QtGui import QPainter, QPen, QPixmap , QImage
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QPoint
import sys
import cv2
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
import torchvision
import torchsummary
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchsummary import summary
from torchvision.models import resnet50
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

app = QtWidgets.QApplication(sys.argv) 
global pic1
window = QtWidgets.QMainWindow()
window.setWindowTitle("HW2")
window.setGeometry(300, 100, 1000, 800) 
loadimage = QtWidgets.QPushButton("Load Image ", window)
loadimage.move(40, 300)

# Create Q1
groupBox1= QtWidgets.QGroupBox("1.Training a CIFAR10 Classifier Using VGG19 with BN ", window)
groupBox1.setFixedSize(250 , 400)
groupBox1.move(180, 30)
pushButton1_1 = QtWidgets.QPushButton("1.1 Show Augmented Images", window)
pushButton1_1.resize(180, 40)
pushButton1_1.move(220, 60)
pushButton1_2 = QtWidgets.QPushButton("1.2 Show Model Structure", window)
pushButton1_2.resize(180, 40)
pushButton1_2.move(220, 125)
pushButton1_3 = QtWidgets.QPushButton("1.3 Show Accuracy and Loss", window)
pushButton1_3.resize(180, 40)
pushButton1_3.move(220, 170)
pushButton1_4 = QtWidgets.QPushButton("1.4 Inference", window)
pushButton1_4.resize(180, 40)
pushButton1_4.move(220, 220)
label1_1 = QtWidgets.QLabel("Predicted = ", window)
label1_1.resize(180, 40)
label1_1.move(220, 290)


#Create Q2
groupBox2= QtWidgets.QGroupBox("2.Training a MNIST Generator Using DcGAN", window)
groupBox2.setFixedSize(230 , 400)
groupBox2.move(450, 30)
pushButton2_1 = QtWidgets.QPushButton("2.1 Show Training Images", window)
pushButton2_1.resize(180, 40)
pushButton2_1.move(470, 60)
pushButton2_2= QtWidgets.QPushButton("2.2 Show Model Structure", window)
pushButton2_2.resize(180, 40)
pushButton2_2.move(470, 120)
pushButton2_3 = QtWidgets.QPushButton("1.3 Show Training Loss", window)
pushButton2_3.resize(180, 40)
pushButton2_3.move(470, 170)
pushButton2_4 = QtWidgets.QPushButton("1.4 Inference", window)
pushButton2_4.resize(180, 40)
pushButton2_4.move(470, 220)

def loadimage_clicked():
    global imageL
    imageL = open_image_using_dialog()
    #cv2.imshow("Image", imageL)
loadimage.clicked.connect(loadimage_clicked)

def open_image_using_dialog():
    options = QFileDialog.Options()
    options |= QFileDialog.ReadOnly
    image_path, _ = QFileDialog.getOpenFileName(None, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tiff);;All Files (*)", options=options)
    print(image_path)
    if image_path:
        image = cv2.imread(image_path)
    else:
        image = None
    return image

data_augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=30)
])

def load_and_augment_images(folder_path):
    augmented_images = []
    image_labels = []
    
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        if os.path.isfile(image_path):  # 確保是檔案
            # 讀取圖片
            image = Image.open(image_path).convert("RGB")
            
            # 套用資料增強
            augmented_image = data_augmentations(image)
            augmented_images.append(augmented_image)

            image_labels.append(os.path.splitext(image_name)[0])  
            
            if len(augmented_images) >= 9:  # 限制只處理前 9 張圖片
                break

    return augmented_images, image_labels

# 顯示增強後的圖片並加上檔名
def plot_augmented_images(augmented_images, image_labels):
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))  # 建立 3x3 的子圖
    
    for i, ax in enumerate(axes.flatten()):
        if i < len(augmented_images):
            ax.imshow(augmented_images[i])
            ax.set_title(image_labels[i], fontsize=10)  # 顯示圖片檔名
            ax.axis("off")  # 移除座標軸
    plt.tight_layout()
    plt.show()

# Load and augment
augmented_images, image_labels = load_and_augment_images(r"C:\Users\user\OneDrive\Desktop\Dataset_CvDl_Hw2\Dataset_CvDl_Hw2\Q1_image\Q1_1")

def pushButton1_1_clicked():
    plot_augmented_images(augmented_images, image_labels)
pushButton1_1.clicked.connect(pushButton1_1_clicked)
# Plot the results

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pushButton1_2_clicked() :
    # 建立 VGG19 模型，並指定 num_classes=10
    model = models.vgg19_bn(num_classes=10)
    
    # 將模型移動到適當的裝置
    model = model.to(device)

    # 使用 torchsummary 顯示模型結構於終端
    print("\n" + "="*30 + " VGG19 Model Structure " + "="*30)
    summary(model, input_size=(3, 32, 32), device=str(device))  # 使用 str(device)
    print("="*80)
pushButton1_2.clicked.connect(pushButton1_2_clicked)

def pushButton1_3_clicked() :
    img = mpimg.imread(r"C:\Users\user\OneDrive\Desktop\Hw1-3 figure.png")
    
    # 创建一个新的figure并显示图片
    plt.figure(figsize=(6, 6))  # 你可以调整figure的大小
    plt.imshow(img)
    plt.axis('off')  # 关闭坐标轴
    plt.show()
pushButton1_3.clicked.connect(pushButton1_3_clicked)

def pushButton1_4_clicked() :
    model = models.vgg19_bn(num_classes=10)
    model.load_state_dict(torch.load('model.pth'))  # Load trained model
    model.eval()
    model.to(device)
pushButton1_4.clicked.connect(pushButton1_4_clicked)
window.show()

sys.exit(app.exec_())
