#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO
import json

class YoloDetectionNode(Node):

    def __init__(self):
        super().__init__('yolo_detection_node')
        
        # Charger le modèle YOLO
        self.model = YOLO('best.pt')
        self.bridge = CvBridge()
        
        # Subscriber : reçoit les images de la caméra
        self.subscription = self.create_subscription(
            Image,
            '/oakd/rgb/preview/image_raw',
            self.camera_callback,
            10
        )
        
        # Publisher : envoie les détections
        self.publisher = self.create_publisher(
            String,
            '/yolo/detections',
            10
        )
        
        self.get_logger().info('Node YOLO démarrée ✅')

    def camera_callback(self, msg):
        # Convertir image ROS → OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        # Appliquer YOLO
        results = self.model(frame, verbose=False)
        
        # Extraire les détections
        detections = []
        for result in results:
            for box in result.boxes:
                detection = {
                    'class': self.model.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].tolist()
                }
                detections.append(detection)
                self.get_logger().info(
                    f"Détecté: {detection['class']} "
                    f"({detection['confidence']:.0%})"
                )
        
        # Publier les détections en JSON
        msg_out = String()
        msg_out.data = json.dumps(detections)
        self.publisher.publish(msg_out)


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
