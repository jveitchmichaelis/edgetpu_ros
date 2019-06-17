#!/usr/bin/env python

import roslib
roslib.load_manifest("edge_tpu")
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

import edgetpu.classification.engine

class tpu_classifier:

    def __init__(self, model, labels, threshold=0.5, device_path=None):
        self.bridge = CvBridge()
        rospy.loginfo("Loading model {}".format(model))
        self.image_sub = rospy.Subscriber("input", Image, self.callback)
        self.threshold = threshold
        self.engine = edgetpu.classification.engine.ClassificationEngine(model, device_path)

        self.load_labels(labels)

        rospy.loginfo("Device path:", engine.device_path())

    def load_labels(self, labels):
        with open(labels, 'r', encoding="utf-8") as f:
            pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
            self.labels = dict((int(k), v) for k, v in pairs)

    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        except CvBridgeError as e:
            rospy.logerr(e)

        results = self.engine.ClassifyWithImage(PIL.Image.fromarray(cv_image), top_k=1, threshold=self.threshold)

        if len(results) > 0:
            try:
                rospy.loginfo("%s %.2f\n%.2fms" % (
                    self.labels[results[0][0]], results[0][1], self.engine.get_inference_time()))
            except:
                rospy.logerr("Error processing results")
                rospy.logerr(results)

def main(args):

    rospy.init_node('classify', anonymous=True)
    
    model_path = rospy.get_param('~model_path')
    label_path = rospy.get_param('~label_path')
    threshold = rospy.get_param('~threshold', default=0.5)
    device_path = rospy.get_param('~device_path', default=None)

    classifier = tpu_classifier(model_path, label_path, threshold, device_path)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == "__main__":
    main(sys.argv)
