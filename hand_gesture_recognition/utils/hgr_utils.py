import copy
import numpy as np
import cv2

def find_hands(input_image, mp, x_lim, y_lim):

        pose = mp["pose"]
        mp_drawing = mp["mp_drawing"]
        mp_pose = mp["mp_pose"]
        mp_drawing_styles = mp["mp_drawing_styles"]
        left_hand_points = mp["left_hand_points"]
        right_hand_points = mp["right_hand_points"]

        hand_left_bounding_box = [0, 0, 0, 0]
        hand_right_bounding_box = [0, 0, 0, 0]

        h, w, _ = input_image.shape
        image = copy.deepcopy(input_image)

        results = pose.process(image)

        annotated_image = image.copy()

        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())


        x_left_points = []
        x_right_points = []
        y_left_points = []
        y_right_points = []

        h, w, _ = image.shape

        hand_left = None
        hand_right = None

        if results.pose_landmarks:
            for id_landmark, landmark in enumerate(results.pose_landmarks.landmark):
                if id_landmark in left_hand_points:
                    x_left_points.append(landmark.x)
                    y_left_points.append(landmark.y)

                if id_landmark in right_hand_points:
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


            hand_left_bounding_box = [l_c[0]-x_lim, l_c[1]-y_lim, l_c[0]+x_lim, l_c[1]+y_lim]
            hand_right_bounding_box = [r_c[0]-x_lim, r_c[1]-y_lim, r_c[0]+x_lim, r_c[1]+y_lim]
            
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

        return hand_left_bounding_box, hand_right_bounding_box, hand_right, hand_left, annotated_image, pose


def take_decision(outputs, preds,thresholds, buffer, cm, min_coef = 0.5):


        if outputs[0][preds] <= thresholds[preds]:
            preds = 4

        buffer.pop(0)
        buffer.append(preds)

        pred = 4


        probability = []
        confidance = []
        coeficients = np.linspace(min_coef, 1.0, num=len(buffer))
        avg_coeficients = sum(coeficients) / len(coeficients)

        for i in range(0, 5):

            prob = 0
            

            for j, prediction in enumerate(buffer):
                prob = prob + (cm[i][prediction] * coeficients[j]) / (100 * len(buffer)) # Weighted Average of probabilities

            probability.append(prob)
            confidance.append(prob / (cm[i][i] * avg_coeficients) * 100)

        pred = probability.index(max(probability))
        confid = confidance[pred]

        return pred, round(confid, 4), buffer
