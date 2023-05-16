#!/usr/bin/env python3
import rospy
from hgr.msg import HandsDetected, HandsClassified
import cv2
import numpy as np
from cv_bridge import CvBridge
import copy
import os
from hand_gesture_recognition.utils.networks import InceptionV3
import torch
import time
from torchvision import transforms
import yaml
from yaml.loader import SafeLoader
from hand_gesture_recognition.utils.hgr_utils import take_decision
from vision_config.vision_definitions import ROOT_DIR
from PIL import Image as PIL_Image


class HandClassificationNode:
    def __init__(self, thresholds, cm,**kargs) -> None:
        # Print the keyword arguments passed to the constructor
        print(kargs)

        # Get the frames per second for classification from a ROS parameter
        fps = rospy.get_param("/hgr/FPS/classification")

        # Initialize a CvBridge instance for ROS image handling
        self.bridge = CvBridge()

        # Subscribe to the "/hgr/hands" topic and set the callback to self.hands_callback
        rospy.Subscriber("/hgr/hands", HandsDetected, self.hands_callback)

        # Create a ROS publisher for the "/hgr/classification" topic
        pub_classification = rospy.Publisher("/hgr/classification", HandsClassified, queue_size=10)

        # Initialize variables for classification
        self.msg = None
        gestures = ["A", "F", "L", "Y", "None"]

        # Initialize variables for classification
        self.thresholds = thresholds

        # Initialize gesture buffers with n_frames "NONE" gestures
        buffer_left = [4] * kargs["n_frames"]
        buffer_right = [4] * kargs["n_frames"]

        # Check if CUDA is available and choose device accordingly
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Training device: ", self.device)

        # Initialize an InceptionV3 model for gesture classification
        self.model = InceptionV3(4, 0.0001, unfreeze_layers=list(np.arange(13, 20)), class_features=2048, device=self.device,
                    con_features=kargs["con_features"])
        self.model.name = kargs["model_name"]

        # Load the trained weights for the model
        trained_weights = torch.load(f'{os.getenv("HOME")}/Datasets/ASL/kinect/results/{self.model.name}/{self.model.name}.pth', map_location=torch.device(self.device))
        self.model.load_state_dict(trained_weights)

        # Put the model into evaluation mode and move it to the chosen device
        self.model.eval()
        self.model.to(self.device)

         # Define a data transform for preprocessing the input images
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.25, 0.25, 0.25])

        self.data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(299),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        print("Waiting!!")
        # Wait for self.msg to be initialized
        while True:
            if self.msg is not None:
                break

        try:
            while not rospy.is_shutdown():

                try:

                    st = time.time()
                    header = self.msg.header
                    im_left = copy.deepcopy(self.msg.hand_left)
                    im_right = copy.deepcopy(self.msg.hand_right)

                    if len(self.msg.hand_left.data) > 0:
                        left_hand = self.bridge.imgmsg_to_cv2(im_left, "rgb8")
                        left_frame = copy.deepcopy(cv2.cvtColor(left_hand, cv2.COLOR_BGR2RGB))
                        outputs, preds = self.predict(left_frame, flip=True)

                        pred_left, confid_left, buffer_left = take_decision(outputs, preds, thresholds, buffer_left, cm, min_coef=kargs["min_coef"])


                    else:
                        pred_left = 4
                        confid_left = 1.0
                    
                    if len(self.msg.hand_right.data) > 0:
                        right_hand = self.bridge.imgmsg_to_cv2(im_right, "rgb8")
                        right_frame = copy.deepcopy(cv2.cvtColor(right_hand, cv2.COLOR_BGR2RGB))
                        outputs, preds = self.predict(right_frame, flip=False)

                        pred_right, confid_right, buffer_right = take_decision(outputs, preds, thresholds, buffer_right, cm, min_coef=kargs["min_coef"])
       
                    
                    else:
                        pred_right = 4 
                        confid_right = 1.0
                    
                    # left = DiagnosticStatus(
                    #     name="left_hand",
                    #     values = [KeyValue(key="classification", value=str(pred_left)), KeyValue(key="confidance", value=str(confid_left))]
                    # )

                    # right = DiagnosticStatus(
                    #     name="right_hand",
                    #     values = [KeyValue(key="classification", value=str(pred_right)), KeyValue(key="confidance", value=str(confid_right))]
                    # )

                    # msg_classification = DiagnosticArray(header = header, status=[left, right])
                    msg_classification = HandsClassified()
                    msg_classification.header = header
                    msg_classification.hand_right.data = gestures[pred_right]
                    msg_classification.hand_left.data = gestures[pred_left]
                    msg_classification.confid_right.data = round(confid_right * 100, 2) 
                    msg_classification.confid_left.data = round(confid_left * 100, 2)
                    while True:

                        if time.time() - st > 1/fps:
                            break
                        
                        time.sleep(1 / (fps * 1000))

                    pub_classification.publish(msg_classification)
                    print(f"CLASSIFICATION Running at {round(1 / (time.time() - st), 2)} FPS")

                except KeyboardInterrupt:
                   break

        except KeyboardInterrupt:
            print("Shutting down")
            cv2.destroyAllWindows()

    def hands_callback(self, msg):
        self.msg = msg   


    def predict(self, hand, flip):
        
        if flip:
            hand = cv2.flip(hand, 1)

        im_norm = self.data_transform(hand).unsqueeze(0)
        im_norm = im_norm.to(self.device)

        with torch.no_grad():   
            outputs, _ = self.model(im_norm)
            _, preds = torch.max(outputs, 1)

        return outputs, preds

    # def take_decision(self, buffer, hand, cm, flip = True):

    #     if flip:
    #         hand = cv2.flip(hand, 1)

    #     im_norm = self.data_transform(hand).unsqueeze(0)
    #     im_norm = im_norm.to(self.device)

    #     with torch.no_grad():   
    #         outputs, _ = self.model(im_norm)
    #         _, preds = torch.max(outputs, 1)

    #     if outputs[0][preds] <= self.thresholds[preds]:
    #         preds = 4

    #     buffer.pop(0)
    #     buffer.append(preds)

    #     pred = 4

    #     # buffer_positives = [i for i in buffer if i != 4] # Removes "None" class

    #     # Decision heuristic 1

    #     # all_equal = all(element == buffer_positives[0] for element in buffer_positives) # Checks if all items are equal to each other

    #     # positives_ratio = len(buffer_positives) / len(buffer)

    #     # if all_equal and positives_ratio > 0.3:
    #     #     pred = buffer_positives[0]

    #     # Decision heuristic 2

    #     # if len(buffer_positives):
    #     #     counter = 0
    #     #     num = buffer_positives[0]
        
    #     #     for i in buffer_positives: # get most frequent classification
    #     #         curr_frequency = buffer_positives.count(i)
    #     #         if(curr_frequency> counter):
    #     #             counter = curr_frequency
    #     #             num = i

    #     #     most_frequent_ratio = counter / len(buffer)
    #     #     most_frequent_positives_ratio = counter / len(buffer_positives)

    #     #     if most_frequent_ratio > t_most_frequent_ratio and most_frequent_positives_ratio > t_most_frequent_positives_ratio:

    #     #         pred = num

    #     # Decision heuristic 3

    #     probability = []
    #     confidance = []

    #     for i in range(0, 5):

    #         prob = 0

    #         for prediction in buffer:
    #             # prob = prob * (cm[i][prediction] / 100 + 0.01) # Multiplicate probabilities

    #             prob = prob + cm[i][prediction] / (100 * len(buffer)) # Average of probabilities

    #         probability.append(prob)
    #         confidance.append(prob / (cm[i][i] / 100))

    #     pred = probability.index(max(probability))
    #     confid = confidance[pred]
    #     # if flip:
    #     #     print("--------------------------------------------")
    #     #     print(f"Buffer: {buffer}")
    #     #     print(f"probability: {probability}")
    #     #     print(f"confidance: {confidance}")
    #     #     print(f"Prediction: {pred}")

    #     return pred, round(confid, 4), buffer


if __name__ == '__main__':

    rospy.init_node("hand_classification", anonymous=True)

    model_name = rospy.get_param("/hgr/model_name", default="InceptionV3")

    with open(f'{ROOT_DIR}/hand_gesture_recognition/config/model/{model_name}.yaml') as f:
        data = yaml.load(f, Loader=SafeLoader)
        print(data)

    with open(f'{ROOT_DIR}/hand_gesture_recognition/config/model/thresholds.yaml') as f:
        t = yaml.load(f, Loader=SafeLoader)
        print(t)

    thresholds = t["thresholds"][data["threshold_choice"]]
    cm = t["confusion_matrix"][data["threshold_choice"]]
    print(thresholds)
    print(cm)


    hc = HandClassificationNode(thresholds, cm, **data)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()