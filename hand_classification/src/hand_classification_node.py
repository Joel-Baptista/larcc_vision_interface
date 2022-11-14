from keras.applications.resnet import ResNet50
import scipy.io
import cv2

# path = ROOT_DIR + "/Datasets/HANDS_dataset/Scripts/"
path = "/home/joel/catkin_ws/src/larcc_vision_interface/Datasets/HANDS_dataset/Scripts/"

mat = scipy.io.loadmat(path + 'base_table.mat')

for key in mat:
    print(key)
    print(mat[key])




