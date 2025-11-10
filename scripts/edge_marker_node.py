#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
edge_marker_node.py
Visualize 3D edge points (PointCloud2) as RViz markers for Task 3.

Params (private, set via roslaunch/rosrun _param:=value):
  ~cloud_topic   (str)  : input PointCloud2 topic [default: "edge_points"]
  ~marker_topic  (str)  : output Marker topic     [default: "edge_markers"]
  ~type          (str)  : "SPHERE_LIST", "POINTS", or "LINE_STRIP" [default: "SPHERE_LIST"]
  ~scale         (float): marker size in meters   [default: 0.005]
  ~alpha         (float): marker alpha (0..1)     [default: 1.0]
  ~color         (list) : [r,g,b] floats 0..1     [default: [0.0, 1.0, 0.0] (green)]
  ~frame_id      (str)  : override frame_id       [default: "" => use cloud.header.frame_id]
  ~lifetime      (float): seconds to keep marker  [default: 0.0 => persistent]

Usage:
  rosrun edge_detection edge_marker_node.py \
      _cloud_topic:=/edge_points \
      _marker_topic:=/edge_markers \
      _type:=SPHERE_LIST _scale:=0.005 _alpha:=1.0

In RViz:
  - Add a "RobotModel" display for the URDF
  - Add a "Marker" display and set Topic to /edge_markers
"""

import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA


class EdgeMarkers:
    def __init__(self):
        self.cloud_topic  = rospy.get_param("~cloud_topic",  "edge_points")
        self.marker_topic = rospy.get_param("~marker_topic", "edge_markers")
        self.marker_type  = rospy.get_param("~type",         "SPHERE_LIST")
        self.scale        = float(rospy.get_param("~scale",  0.005))
        self.alpha        = float(rospy.get_param("~alpha",  1.0))
        self.color        = rospy.get_param("~color",        [0.0, 1.0, 0.0])
        self.frame_override = rospy.get_param("~frame_id",   "")
        self.lifetime_s   = float(rospy.get_param("~lifetime", 0.0))

        # Map string type to Marker enum
        self._type_map = {
            "SPHERE_LIST": Marker.SPHERE_LIST,
            "POINTS":      Marker.POINTS,
            "LINE_STRIP":  Marker.LINE_STRIP,
        }
        self._marker_type = self._type_map.get(self.marker_type.upper(), Marker.SPHERE_LIST)

        self.pub = rospy.Publisher(self.marker_topic, Marker, queue_size=1)
        self.sub = rospy.Subscriber(self.cloud_topic, PointCloud2, self.cb, queue_size=5)

        rospy.loginfo("edge_marker_node: listening on '%s', publishing '%s' as %s",
                      self.cloud_topic, self.marker_topic, self.marker_type)

    def cb(self, cloud_msg: PointCloud2):
        marker = Marker()
        marker.header.stamp = cloud_msg.header.stamp
        marker.header.frame_id = self.frame_override or cloud_msg.header.frame_id
        marker.ns = "edge_points"
        marker.id = 0
        marker.type = self._marker_type
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration.from_sec(self.lifetime_s)

        # scale: for POINTS use x/y, for SPHERE_LIST use x/y/z (z ignored by RViz for POINTS)
        marker.scale.x = self.scale
        marker.scale.y = self.scale
        marker.scale.z = self.scale

        # single color for all points (POINTS can also take per-point colors; not required)
        r, g, b = (float(self.color[0]), float(self.color[1]), float(self.color[2]))
        marker.color = ColorRGBA(r=r, g=g, b=b, a=self.alpha)

        # Convert cloud -> list of Points
        for x, y, z in point_cloud2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
            marker.points.append(Point(x=float(x), y=float(y), z=float(z)))

        # For LINE_STRIP, if you want a closed loop, append first point at end:
        if self._marker_type == Marker.LINE_STRIP and len(marker.points) > 2:
            marker.points.append(marker.points[0])

        self.pub.publish(marker)


def main():
    rospy.init_node("edge_marker_node")
    _ = EdgeMarkers()
    rospy.spin()


if __name__ == "__main__":
    main()

