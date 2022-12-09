#!/usr/bin/env python3
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

if __name__ == '__main__':

    rospy.init_node("home_test", anonymous=True)

    vid = cv2.VideoCapture(0)

    pub_image = rospy.Publisher("/camera/rgb/image_raw", Image, queue_size=10)
    bridge = CvBridge()

    while True:

        ret, frame = vid.read()

        # cv2.imshow('frame', frame)

        pub_image.publish(bridge.cv2_to_imgmsg(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), "rgb8"))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()
