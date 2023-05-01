#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge
import copy
import mediapipe as mp
import os
from hand_gesture_recognition.utils.networks import InceptionV3
import torch
import time
from torchvision import transforms


class HandGestureRecognition:
    def __init__(self) -> None:

        # Get initial data from rosparams

        # image_topic = rospy.get_param("/hgr/image_topic", default="/camera/color/image_raw")
        image_topic = rospy.get_param("/hgr/image_topic", default="/camera/rgb/image_raw")
        model_name = rospy.get_param("/hgr/model_name", default="InceptionV3")
        con_features = rospy.get_param("/hgr/con_features", default=16)
        self.thresholds = rospy.get_param("/hgr/tresholds", default=[0, 0, 0, 0])
        
        # Initializations for MediaPipe to detect keypoints
        self.left_hand_points = (16, 18, 20, 22)
        self.right_hand_points = (15, 17, 19, 21)

        self.bridge = CvBridge()

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False,
                                      model_complexity=2,
                                      enable_segmentation=False,
                                      min_detection_confidence=0.7)

        rospy.Subscriber(image_topic, Image, self.image_callback)

        self.cv_image = None
        self.bridge = CvBridge()

        # Initialize variables for classification
        org = (0, 90)
        thickness = 1
        font = 1
        gestures = ["A", "F", "L", "Y", "NONE"]

        buffer_left = [4] * 5 # Initializes the buffer with 5 "NONE" gestures
        buffer_right = [4] * 10

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Training device: ", self.device)

        self.model = InceptionV3(4, 0.0001, unfreeze_layers=list(np.arange(13, 20)), class_features=2048, device=self.device,
                    con_features=con_features)
        self.model.name = model_name

        trained_weights = torch.load(f'{os.getenv("HOME")}/Datasets/ASL/kinect/results/{self.model.name}/{self.model.name}.pth', map_location=torch.device(self.device))
        self.model.load_state_dict(trained_weights)

        self.model.eval()

        self.model.to(self.device)

        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.25, 0.25, 0.25])

        self.data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(299),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Left Hand", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Right Hand", cv2.WINDOW_NORMAL)


        print("Waiting!!")
        while True:
            if self.cv_image is not None:
                break

        while True:

            hand_right, hand_left, keypoints_image = self.find_hands(copy.deepcopy(self.cv_image), x_lim=50, y_lim=50)

            if hand_left is not None:
                
                pred_left, buffer_left = self.take_decision(buffer_left, hand_left, flip=True)

            else:

                pred_left = 4
                # hand_left = np.zeros((100, 100, 3))

            if hand_right is not None:
                
                pred_right, buffer_right = self.take_decision(buffer_right, hand_right, flip=False)

            else:

                pred_right = 4
                # hand_right = np.zeros((100, 100, 3))



            # if hand_left is not None and hand_right is not None:

            #     im1 = data_transform(cv2.flip(hand_left, 1))
            #     im2 = data_transform(hand_right)

            #     im_array_norm = torch.cat((im1.unsqueeze(0), im2.unsqueeze(0)), 0)
            #     im_array_norm = im_array_norm.to(device)

                
            #     with torch.no_grad():
            #         outputs, _ = model(im_array_norm)
            #         _, preds = torch.max(outputs, 1)

                
            #     if outputs[0][preds[0]] <= thresholds[preds[0]]:
            #         preds[0] = 4

            #     if outputs[1][preds[1]] <= thresholds[preds[1]]:
            #         preds[1] = 4


            #     buffer_left.pop(0)
            #     buffer_left.append(preds[0])

            #     buffer_right.pop(0)
            #     buffer_right.append(preds[1])

            #     pred_left = self.take_decision(buffer_left)
            #     pred_right = self.take_decision(buffer_right)

            hand_left = cv2.putText(hand_left, gestures[pred_left], org, cv2.FONT_HERSHEY_SIMPLEX,
                            font, (255, 0, 0), thickness, cv2.LINE_AA)

            hand_right = cv2.putText(hand_right, gestures[pred_right], org, cv2.FONT_HERSHEY_SIMPLEX,
                                    font, (255, 0, 0), thickness, cv2.LINE_AA)
                

            cv2.imshow('Original Image', cv2.cvtColor(keypoints_image, cv2.COLOR_BGR2RGB))

            if hand_left is not None:
                cv2.imshow('Left Hand',  cv2.cvtColor(hand_left, cv2.COLOR_BGR2RGB))
            
            if hand_right is not None:
                cv2.imshow('Right Hand',  cv2.cvtColor(hand_right, cv2.COLOR_BGR2RGB))

            key = cv2.waitKey(1)

            if key == 113:
                break

        cv2.destroyAllWindows()

    
    def take_decision(self, buffer, hand, flip = True):

        if flip:
            hand = cv2.flip(hand, 1)

        im_norm = self.data_transform(hand).unsqueeze(0)
        im_norm = im_norm.to(self.device)

        with torch.no_grad():   
            outputs, _ = self.model(im_norm)
            _, preds = torch.max(outputs, 1)

        if outputs[0][preds] <= self.thresholds[preds]:
            preds = 4

        buffer.pop(0)
        buffer.append(preds)

        pred = 4

        buffer_positives = [i for i in buffer if i != 4] # Removes "None" class
        
        # all_equal = all(element == buffer_positives[0] for element in buffer_positives) # Checks if all items are equal to each other

        # positives_ratio = len(buffer_positives) / len(buffer)

        # if all_equal and positives_ratio > 0.2:
        #     pred = buffer_positives[0]

        #TODO Put all this values in configuration files

        if len(buffer_positives):
            counter = 0
            num = buffer_positives[0]
        
            for i in buffer_positives: # get most frequent classification
                curr_frequency = buffer_positives.count(i)
                if(curr_frequency> counter):
                    counter = curr_frequency
                    num = i

            most_frequent_ratio = counter / len(buffer)
            most_frequent_positives_ratio = counter / len(buffer_positives)

            if most_frequent_ratio > 0.3 and most_frequent_positives_ratio > 0.8:

                pred = num

        return pred, buffer
    
    # def take_decision(buffer):

    #     pred = 4

    #     buffer_positives = [i for i in buffer if i != 4] # Removes "None" class
        
    #     all_equal = all(element == buffer_positives[0] for element in buffer_positives) # Checks if all items are equal to each other
        
    #     positives_ratio = len(buffer_positives) / len(buffer)

    #     if all_equal and positives_ratio > 0.5:

    #         pred = buffer_positives[0]

    #     return pred


    def image_callback(self, msg):
        
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")

    def find_hands(self, input_image, x_lim, y_lim):

        h, w, _ = input_image.shape
        image = copy.deepcopy(input_image)

        results = self.pose.process(image)

        annotated_image = image.copy()

        self.mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())


        x_left_points = []
        x_right_points = []
        y_left_points = []
        y_right_points = []

        h, w, _ = image.shape

        hand_left = None
        hand_right = None

        if results.pose_landmarks:
            for id_landmark, landmark in enumerate(results.pose_landmarks.landmark):
                if id_landmark in self.left_hand_points:
                    x_left_points.append(landmark.x)
                    y_left_points.append(landmark.y)

                if id_landmark in self.right_hand_points:
                    x_right_points.append(landmark.x)
                    y_right_points.append(landmark.y)

            l_c = [int(np.mean(x_left_points) * w), int(np.mean(y_left_points) * h)]
            r_c = [int(np.mean(x_right_points) * w), int(np.mean(y_right_points) * h)]


            if l_c[0] < x_lim:
                l_c[0] = x_lim
            if l_c[1] < y_lim:
                l_c[1] = y_lim
            if r_c[0] < x_lim:
                r_c[0] = x_lim
            if r_c[1] < y_lim:
                r_c[1] = y_lim

            hand_left = input_image[l_c[1]-y_lim:l_c[1]+y_lim,
                                                        l_c[0]-x_lim:l_c[0]+x_lim]

            hand_right = input_image[r_c[1]-y_lim:r_c[1]+y_lim,
                                                         r_c[0]-x_lim:r_c[0]+x_lim]

            left_start_point = (l_c[0]-x_lim, l_c[1]-y_lim)
            left_end_point = (l_c[0]+x_lim, l_c[1]+y_lim)

            right_start_point = (r_c[0]-x_lim, r_c[1]-y_lim)
            right_end_point = (r_c[0]+x_lim, r_c[1]+y_lim)

            cv2.rectangle(annotated_image, left_start_point, left_end_point, (255, 0, 0), 2)
            cv2.rectangle(annotated_image, right_start_point, right_end_point, (255, 0, 0), 2)

        if np.array(hand_left).shape != (2*x_lim, 2*y_lim, 3):
            hand_left = None

        if np.array(hand_right).shape != (2*x_lim, 2*y_lim, 3):
            hand_right = None

        return hand_right, hand_left, annotated_image



if __name__=="__main__":

    rospy.init_node("hand_gesture_recognition", anonymous=True)
    hd = HandGestureRecognition()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()
