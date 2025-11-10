#!/usr/bin/env python3
import rospy, cv2, numpy as np
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
from edge_detection.edge_detector import detect_outer_box

class EdgeNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.rgb_sub = rospy.Subscriber(rospy.get_param("~rgb_topic", "/camera/color/image_raw"),
                                        Image, self.rgb_cb, queue_size=1)
        self.depth_sub = rospy.Subscriber(rospy.get_param("~depth_topic", "/camera/depth/image_rect_raw"),
                                          Image, self.depth_cb, queue_size=1)
        self.info_sub = rospy.Subscriber(rospy.get_param("~info_topic", "/camera/color/camera_info"),
                                         CameraInfo, self.info_cb, queue_size=1)
        self.edges_pub = rospy.Publisher("edges", Image, queue_size=1)
        self.cloud_pub = rospy.Publisher("edge_points", PointCloud2, queue_size=1)

        self.depth_msg = None
        self.K = None  # fx, fy, cx, cy

    def info_cb(self, msg):
        self.K = (msg.K[0], msg.K[4], msg.K[2], msg.K[5])  # fx, fy, cx, cy
        self.frame_id = msg.header.frame_id

    def depth_cb(self, msg):
        self.depth_msg = msg

    def rgb_cb(self, msg):
        if self.K is None or self.depth_msg is None:
            return

        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        overlay, quad, edges = detect_outer_box(bgr, draw_all_edges=False)

        # Publish overlay image
        out_img = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
        out_img.header = msg.header
        self.edges_pub.publish(out_img)

        # Build a thin set of edge samples along the quad boundary
        if not quad:
            return
        h, w = edges.shape[:2]
        fx, fy, cx, cy = self.K

        # get depth as CV_16UC1 or CV_32FC1
        depth = self.bridge.imgmsg_to_cv2(self.depth_msg, desired_encoding="passthrough")
        depth_is_16u = depth.dtype == np.uint16

        # sample points densely along the 4 lines
        def sample_line(p1, p2, step=1.0):
            p1 = np.array(p1, np.float32); p2 = np.array(p2, np.float32)
            L = np.linalg.norm(p2-p1)
            n = max(2, int(L/step))
            alphas = np.linspace(0, 1, n)
            pts = (1-alphas)[:,None]*p1 + alphas[:,None]*p2
            return np.round(pts).astype(int)

        edge_pixels = []
        for i in range(4):
            p1 = quad[i]; p2 = quad[(i+1)%4]
            pts = sample_line(p1, p2, step=1.0)
            for u,v in pts:
                if 0<=u<w and 0<=v<h:
                    edge_pixels.append((u,v))

        # Back-project to XYZ
        xyz = []
        for u,v in edge_pixels:
            z = float(depth[v,u])
            if depth_is_16u:
                z *= 0.001  # mm→m
            if not np.isfinite(z) or z <= 0:
                continue
            x = ( (u - cx) * z ) / fx
            y = ( (v - cy) * z ) / fy
            xyz.append((x,y,z))

        if not xyz:
            return

        # Publish PointCloud2
        fields = [
            PointField('x', 0,  PointField.FLOAT32, 1),
            PointField('y', 4,  PointField.FLOAT32, 1),
            PointField('z', 8,  PointField.FLOAT32, 1),
        ]
        cloud = pc2.create_cloud(msg.header, fields, xyz)
        cloud.header.frame_id = getattr(self, "frame_id", msg.header.frame_id)
        self.cloud_pub.publish(cloud)

if __name__ == "__main__":
    rospy.init_node("edge_node")
    EdgeNode()
    rospy.spin()

