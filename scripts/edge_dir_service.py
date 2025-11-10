#!/usr/bin/env python3
import rospy, cv2
from pathlib import Path
from edge_detection.srv import DetectEdgesFromDir, DetectEdgesFromDirResponse
from edge_detector import detect_outer_box

def handle(req):
    p = Path(req.dir_path)
    if not p.exists():
        return DetectEdgesFromDirResponse(ok=False, processed=0)
    count = 0
    out_dir = p / "edges_out"
    out_dir.mkdir(exist_ok=True, parents=True)
    exts = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
    for img_path in p.iterdir():
        if not (img_path.is_file() and img_path.suffix.lower() in exts):
            continue
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None: continue
        overlay, _, _ = detect_outer_box(bgr, draw_all_edges=False)
        cv2.imwrite(str(out_dir / f"{img_path.stem}_edges{img_path.suffix}"), overlay)
        count += 1
    return DetectEdgesFromDirResponse(ok=True, processed=count)

if __name__ == "__main__":
    rospy.init_node("edge_dir_service")
    srv = rospy.Service("detect_edges_from_dir", DetectEdgesFromDir, handle)
    rospy.spin()

