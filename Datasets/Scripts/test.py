import cv2
from vision_config.vision_definitions import ROOT_DIR
import tensorflow as tf
import numpy as np
from hand_detection.src.pose_detection import PoseDetection
import copy

img1 = cv2.imread(ROOT_DIR + "/Datasets/ASL/train/A/A1.jpg")
img3 = cv2.imread(ROOT_DIR + "/Datasets/ASL/train/A/A1675.jpg")
img4 = cv2.imread(ROOT_DIR + "/Datasets/ASL/train/A/A500.jpg")
img2 = cv2.imread(ROOT_DIR + "/Datasets/Larcc_dataset/A/image125.png")

pd = PoseDetection()

pd.cv_image = copy.deepcopy(img2)
pd.detect_pose()
pd.find_hands(x_lim=100, y_lim=100)

# print(img1)
# print("=====================================================================")
# print(img2)

# preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
preprocess_input = tf.keras.applications.resnet50.preprocess_input

feature_inputs = tf.keras.Input(shape=(200, 200, 3))
# x_f = data_augmentation(feature_inputs)
x_f = preprocess_input(feature_inputs)


extractor = tf.keras.Model(feature_inputs, x_f)
print(img1.shape)

extractor.summary()

predictions = extractor.predict(x=np.array([img1, pd.cv_image_detected_left, img3, img4]), verbose=2)

print(predictions.shape)
print(predictions[0])
cv2.imshow("Original img1", img1)
cv2.imshow("test img1", np.array(predictions[0]))

cv2.imshow("Original img2", pd.cv_image_detected_left)
cv2.imshow("test img2", np.array(predictions[1]))

cv2.imshow("Original img3", img3)
cv2.imshow("test img3", np.array(predictions[2]))

cv2.imshow("Original img43", img4)
cv2.imshow("test img4", np.array(predictions[3]))
cv2.waitKey()
