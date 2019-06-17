#!/usr/bin/env python

import roslib
roslib.load_manifest("boson")
import sys
import rospy
import cv2
import time
from std_msgs.msg import String
from sensor_msgs.msg import Image
#from object_detection_msgs.msgs import RecognizedObject
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import PIL

import edgetpu.detection.engine

class tpu_detector:

    def __init__(self, model, labels, threshold=0.5, device_path=None):
        self.bridge = CvBridge()
        rospy.loginfo("Loading model {}".format(model))
        self.image_sub = rospy.Subscriber("input", Image, self.callback)
        self.threshold = threshold
        self.engine = edgetpu.detection.engine.DetectionEngine(model, device_path)

        self.load_labels(labels)

    def load_labels(self, labels):
        with open(labels, 'r', encoding="utf-8") as f:
            pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
            self.labels = dict((int(k), v) for k, v in pairs)

    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        except CvBridgeError as e:
            rospy.logerr(e)

        results = self.engine.DetectWithImage(PIL.Image.fromarray(cv_image), top_k=1, threshold=self.threshold, keep_aspect_ratio=True, relative_coord=True)

        for detection in results:
            try:
                rospy.loginfo(self.labels[detection.label_id])
                rospy.loginfo(detection.bounding_box)
                
            except:
                rospy.logerr("Error processing results")
                rospy.logerr(results)
        
        rospy.logdebug("%.2f ms" % self.engine.get_inference_time())

def main(args):

    rospy.init_node('detect', anonymous=True)
    
    model_path = rospy.get_param('~model_path')
    label_path = rospy.get_param('~label_path')
    threshold = rospy.get_param('~threshold', default=0.5)
    device_path = rospy.get_param('~device_path', default=None)

    detector = tpu_detector(model_path, label_path, threshold, device_path)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == "__main__":
    main(sys.argv)
