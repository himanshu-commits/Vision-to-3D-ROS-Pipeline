#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

# -----------------------
# Helpers
# -----------------------

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def ensure_odd(k: int) -> int:
    return k if (k % 2) == 1 else (k + 1)

def clahe_gray(bgr, clip=3.0, grid=8):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return clahe.apply(gray)

def auto_canny_thresholds(gray, low_scale=0.66, high_scale=1.33):
    # Median from histogram (fast & deterministic)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    cum = np.cumsum(hist)
    half = gray.size / 2.0
    med = int(np.searchsorted(cum, half))
    low  = max(0, int(low_scale  * med))
    high = min(255, int(high_scale * med))
    return low, high

def order_clockwise(pts):
    pts = np.asarray(pts, dtype=np.float32)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    idx = np.argsort(ang)
    return pts[idx].astype(int)

def gather_inputs(path: Path, recursive: bool):
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    if recursive:
        return [p for p in path.rglob("*") if p.suffix.lower() in IMG_EXTS]
    return [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]

# -----------------------
# Core detection
# -----------------------

def detect_outer_box(bgr,
                     blur_ksize=5,
                     clahe_clip=3.0,
                     clahe_grid=8,
                     canny_low_scale=0.66,
                     canny_high_scale=1.33,
                     morph_ksize=5,
                     draw_edge_overlay=True,
                     **kwargs):
    """
    Returns:
      overlay (BGR): original with green outer box (and optional thin green edge map)
      quad (list[(x,y)]): 4 points (clockwise) of the outer box; [] if not found
      edges (uint8): canny edge map (0/255)
    Notes:
      - Accepts alias 'draw_all_edges' (bool) to match ROS node call signature.
    """
    # Alias handling so ROS node can call draw_all_edges=False
    if 'draw_all_edges' in kwargs:
        draw_edge_overlay = bool(kwargs['draw_all_edges'])

    if bgr is None or bgr.size == 0:
        return None, [], np.zeros((1,1), np.uint8)

    blur_ksize = ensure_odd(int(blur_ksize))
    morph_ksize = ensure_odd(int(morph_ksize))

    # 1) CLAHE gray
    gray = clahe_gray(bgr, clip=clahe_clip, grid=clahe_grid)

    # 2) Blur
    gray_blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # 3) Auto-Canny
    low, high = auto_canny_thresholds(gray_blur, canny_low_scale, canny_high_scale)
    edges = cv2.Canny(gray_blur, low, high, apertureSize=3, L2gradient=True)

    # 4) Closing to seal gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_ksize, morph_ksize))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 5) Largest external contour (treat board as one component)
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = bgr.copy()
    if not contours:
        if draw_edge_overlay:
            overlay[edges.astype(bool)] = (0, 255, 0)
        return overlay, [], edges

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 50:
        if draw_edge_overlay:
            overlay[edges.astype(bool)] = (0, 255, 0)
        return overlay, [], edges

    # 6) Background-aligned outer box via minAreaRect
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect).astype(int)
    quad = order_clockwise(box)

    # 7) Draw
    if draw_edge_overlay:
        overlay[edges.astype(bool)] = (0, 255, 0)  # thin green edge map
    for i in range(4):
        p1 = tuple(map(int, quad[i]))
        p2 = tuple(map(int, quad[(i + 1) % 4]))
        cv2.line(overlay, p1, p2, (0, 255, 0), 3, cv2.LINE_AA)

    return overlay, quad.tolist(), edges

# -----------------------
# CLI processing (folder in → folder out)
# -----------------------

def process_one(in_path: Path, out_dir: Path, suffix="_edges", draw_edge_overlay=True) -> bool:
    bgr = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"[WARN] Could not read: {in_path}", file=sys.stderr)
        return False

    result = detect_outer_box(bgr, draw_edge_overlay=draw_edge_overlay)
    # Backward/forward compatible unpack
    if isinstance(result, tuple) and len(result) == 3:
        overlay, quad, _edges = result
    else:
        overlay, quad = result
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{in_path.stem}{suffix}{in_path.suffix}"

    ok = cv2.imwrite(str(out_path), overlay)
    if not ok:
        print(f"[WARN] Failed to write: {out_path}", file=sys.stderr)
        return False

    print(f"[OK] {in_path.name} → {out_path.name}   quad={quad}")
    return True

def parse_args():
    ap = argparse.ArgumentParser(description="Checkerboard outer-edge detector (folder in → folder out)")
    ap.add_argument("--input", required=True, help="Image file or folder")
    ap.add_argument("--output", required=True, help="Output folder")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders when input is a folder")
    ap.add_argument("--no-edge-overlay", action="store_true", help="Do NOT paint full edge map; draw box only")
    return ap.parse_args()

def main():
    args = parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.output)

    files = gather_inputs(in_path, recursive=args.recursive)
    if not files:
        print(f"[ERROR] No valid images found for: {in_path}", file=sys.stderr)
        sys.exit(2)

    ok_all = True
    for p in files:
        ok = process_one(p, out_dir, draw_edge_overlay=not args.no_edge_overlay)
        ok_all = ok_all and ok

    sys.exit(0 if ok_all else 1)

if __name__ == "__main__":
    main()

