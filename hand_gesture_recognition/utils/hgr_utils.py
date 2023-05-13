import copy
import numpy as np
import cv2

def find_hands(input_image, mp, x_lim, y_lim):
    """
    Finds the bounding boxes and images of the left and right hands in the input image.
    
    Args:
    input_image (numpy array): The input image as a numpy array.
    mp (dict): A dictionary containing information for the MediaPipe library.
    x_lim (int): The half-length of the bounding box in the x-axis direction.
    y_lim (int): The half-length of the bounding box in the y-axis direction.
    
    Returns:
    hand_left_bounding_box (list): A list of integers representing the bounding box of the left hand in the format [x_min, y_min, x_max, y_max].
    hand_right_bounding_box (list): A list of integers representing the bounding box of the right hand in the format [x_min, y_min, x_max, y_max].
    hand_right (numpy array): A numpy array representing the image of the right hand.
    hand_left (numpy array): A numpy array representing the image of the left hand.
    annotated_image (numpy array): A numpy array representing the annotated input image with bounding boxes drawn around the hands.
    pose (MediaPipe module): The MediaPipe Pose module.
    """

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

    # Process the image using the MediaPipe Pose module.
    results = pose.process(image)

    annotated_image = image.copy()

    # Draw the landmarks of the detected pose on the annotated image.
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

    # If pose landmarks were detected, extract the x and y coordinates of the left and right hand landmarks.
    if results.pose_landmarks:
        for id_landmark, landmark in enumerate(results.pose_landmarks.landmark):
            if id_landmark in left_hand_points:
                x_left_points.append(landmark.x)
                y_left_points.append(landmark.y)

            if id_landmark in right_hand_points:
                x_right_points.append(landmark.x)
                y_right_points.append(landmark.y)

        # Calculate the center points of the left and right hands.
        l_c = [int(np.mean(x_left_points) * w), int(np.mean(y_left_points) * h)]
        r_c = [int(np.mean(x_right_points) * w), int(np.mean(y_right_points) * h)]

        # Adjust the center points if they are out of bounds.
        if l_c[0] < x_lim:
            l_c[0] = x_lim
        if l_c[1] < y_lim:
            l_c[1] = y_lim
        if r_c[0] < x_lim:
            r_c[0] = x_lim
        if r_c[1] < y_lim:
            r_c[1] = y_lim

        # Acquire images from the region of interest.
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
    """
    Takes a decision based on the current prediction and the prediction history.

    Args:
        outputs (list): List of predicted probabilities for each class.
        preds (int): Index of the current predicted class.
        thresholds (list): List of threshold values for each class.
        buffer (list): List of previous predictions.
        cm (ndarray): Confusion matrix of the model.
        min_coef (float): Minimum value for the coefficients used in the weighted average. Default is 0.5.

    Returns:
        pred (int): The predicted class.
        confid (float): The confidence of the prediction.
        buffer (list): The updated buffer.
    """

    # If the predicted probability for the current class is below its threshold, set the predicted class to 'None'.
    if outputs[0][preds] <= thresholds[preds]:
        preds = 4

    buffer.pop(0)
    buffer.append(preds)

     # Initialize variables.
    pred = 4
    probability = []
    confidance = []
    coeficients = np.linspace(min_coef, 1.0, num=len(buffer))
    avg_coeficients = sum(coeficients) / len(coeficients)

    # Compute the weighted average of the probabilities and the confidence for each class.
    for i in range(0, 5):
        prob = 0
        for j, prediction in enumerate(buffer):
            prob = prob + (cm[i][prediction] * coeficients[j]) / (100 * len(buffer)) 

        probability.append(prob)
        confidance.append(prob / (cm[i][i] * avg_coeficients) * 100)

    # Set the predicted class to the class with the highest probability.
    pred = probability.index(max(probability))
    confid = confidance[pred]

    return pred, round(confid, 4), buffer
