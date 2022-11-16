#!/usr/bin/env python3
import os
from hand_detection.src.pose_detection import PoseDetection
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import keras
import numpy as np
import cv2
from itertools import product
import matplotlib.pyplot as plt
from vision_config.vision_definitions import ROOT_DIR


def plot_confusion_matrix_percentage(confusion_matrix, display_labels=None, cmap="viridis",
                                     xticks_rotation="horizontal", title="Confusion Matrix"):
    colorbar = True
    im_kw = None
    fig, ax = plt.subplots()
    cm = confusion_matrix
    n_classes = cm.shape[0]

    default_im_kw = dict(interpolation="nearest", cmap=cmap)
    im_kw = im_kw or {}
    im_kw = {**default_im_kw, **im_kw}

    im_ = ax.imshow(cm, **im_kw)
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(1.0)

    text_ = np.empty_like(cm, dtype=object)

    # print text with appropriate color depending on background
    thresh = (cm.max() + cm.min()) / 2.0

    for i, j in product(range(n_classes), range(n_classes)):
        color = cmap_max if cm[i, j] < thresh else cmap_min
        # text_cm = format(cm[i, j], ".1f") + " %"
        text_cm = format(cm[i, j], ".1f")
        text_[i, j] = ax.text(
            j, i, text_cm, ha="center", va="center", color=color
        )

    if display_labels is None:
        display_labels = np.arange(n_classes)
    else:
        display_labels = display_labels
    if colorbar:
        fig.colorbar(im_, ax=ax)
    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=display_labels,
        yticklabels=display_labels,
        ylabel="True label",
        xlabel="Predicted label",
    )

    ax.set_ylim((n_classes - 0.5, -0.5))
    fig.suptitle(title)
    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)




subject = 1
gesture = 3

# dataset_path = ROOT_DIR + "/Datasets/HANDS_dataset"
# data_path = f"/Subject{subject}/Processed/G{gesture}/"

dataset_path = ROOT_DIR + "/Datasets/Joel_v2/Processed/"
data_path = f""

model = keras.models.load_model("myModel_second")
count_false = 0
count_true = 0

ground_truth = []
predictions = []

gestures = ['one finger', 'open hand', 'fist']

for gesture in range(1, 4):
    res = os.listdir(dataset_path + f"G{gesture}")
    true_value = 0
    for file in res:
        im = cv2.imread(dataset_path + f"G{gesture}/" + file)
        # pd = PoseDetection()
        #
        # pd.cv_image = im
        # pd.detect_pose()
        # pd.find_hands()

        # cv2.imshow('Left Hand', pd.cv_image_detected_left)
        # cv2.imshow('Right Hand', pd.cv_image_detected_right)

        im_array = np.asarray([im])

        prediction = model.predict(x=im_array, verbose=2)

        predictions.append(gestures[np.argmax(prediction)])
        ground_truth.append(gestures[gesture - 1])

        print(gesture - 1)
        print(np.argmax(prediction))
        if gesture == (np.argmax(prediction) + 1):
            count_true += 1
        else:
            count_false += 1


cm = confusion_matrix(ground_truth, predictions, labels=gestures)
blues = plt.cm.Blues
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gestures)
disp.plot(cmap=blues)


print(f"Accuracy: {round(count_true / (count_false + count_true) * 100, 2)}%")
print(f"Tested with: {count_false + count_true}")

plt.show()
