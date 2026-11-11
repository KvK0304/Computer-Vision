import argparse, csv, glob, json, os
import cv2
import numpy as np

BORDER_RATIO = 0.196      
RING_MIN, INNER_MIN = 0.65, 0.10   
BLUR_SHARPNESS = 0.6               


def warm_candidates(img):
    """Warm-colour candidate contours + the denoised grey image used by all later tests."""
    den = cv2.bilateralFilter(img, 5, 50, 50)
    L, a, b = cv2.split(cv2.cvtColor(den, cv2.COLOR_BGR2Lab))
    L = cv2.createCLAHE(2.0, (8, 8)).apply(L)                                  
    hsv = cv2.cvtColor(cv2.cvtColor(cv2.merge((L, a, b)), cv2.COLOR_Lab2BGR), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (5, 80, 60), (35, 255, 255))                         
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c, h in zip(cnts, hier[0]) if h[3] == -1]                    
    return cv2.cvtColor(den, cv2.COLOR_BGR2GRAY), cnts


def measure(gray, contour):
    """Structural evidence for one candidate, or None if it is not even roughly the right shape/size."""
    H, W = gray.shape
    area = cv2.contourArea(contour)
    if area < 80 or area > 0.4 * H * W:
        return None
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    ar = max(rw, rh) / max(1.0, min(rw, rh))
    if min(rw, rh) < 8 or ar > 2.2:                                          
        return None
    blob = np.zeros((H, W), np.uint8)
    cv2.drawContours(blob, [contour], -1, 1, -1)
    g = gray.astype(np.float32)
    bright = np.percentile(g[blob > 0], 90)
    core = ((blob > 0) & (g >= 0.7 * bright)).astype(np.uint8)                   
    n, lab, st, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    if n < 2:
        return None
    core = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    cc, _ = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(cc, key=cv2.contourArea))                          
    hm = np.zeros((H, W), np.uint8)
    cv2.drawContours(hm, [hull], -1, 1, -1)
    side = np.sqrt(float(hm.sum()))
    t = max(3.0, BORDER_RATIO * side)                                          
    fill = float(np.median(g[core > 0]))
    dist = cv2.distanceTransform((hm == 0).astype(np.uint8), cv2.DIST_L2, 3)
    ring = (dist > 1.5) & (dist <= 1.3 * t)                                      
    expected = (t - 1.5) / (1.3 * t - 1.5)           
    ring_dark = min(1.0, float((g[ring] < 0.5 * fill).mean()) / expected) if ring.any() else 0.0
    ring_relaxed = min(1.0, float((g[ring] < 0.65 * fill).mean()) / expected) if ring.any() else 0.0  
    inner_dark = float((g[hm > 0] < 0.55 * fill).mean())
    x, y, w, h = cv2.boundingRect(hull)
   
    roi = g[max(0, y - 5):y + h + 5, max(0, x - 5):x + w + 5]
    gx, gy = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3) / 8.0, cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    sharp = 2 * float(np.percentile(np.hypot(gx, gy), 99)) / max(1.0, float(np.percentile(roi, 95) - np.percentile(roi, 5)))
    
    cx, cy = [int(v) for v in np.mean(hull[:, 0, :], axis=0)]
    r = max(2, int(0.08 * side))
    centre = float(g[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1].min())
    centre_dark = centre < 0.7 * fill
    
    frame = ring & (g < 0.5 * fill)
    ys, xs = np.where((hm > 0) | frame)
    box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    touches = x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1
    return dict(ring=ring_dark, inner=inner_dark, ring_relaxed=ring_relaxed, sharp=sharp, centre_dark=centre_dark,
                side=side, ar=ar, box=box, touches_border=touches, fill_box=[x, y, x + w, y + h])


def classify(m):
    """-> 'tag' | 'tag_blurred' | 'near_miss' | None"""
    if m["ring"] >= RING_MIN and m["inner"] >= INNER_MIN:
        return "tag"
    if m["sharp"] < BLUR_SHARPNESS and m["ring_relaxed"] >= RING_MIN and m["centre_dark"]:   #
        return "tag_blurred"
    if (m["ring"] >= RING_MIN) != (m["inner"] >= INNER_MIN):                   
        return "near_miss"
    return None


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / float((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-9)


def detect(img):
    gray, cnts = warm_candidates(img)
    found, near = [], []
    for c in cnts:
        m = measure(gray, c)
        if m is None:
            continue
        kind = classify(m)
        if kind in ("tag", "tag_blurred"):
            notes = []
            if kind == "tag_blurred":
                notes.append("blurred: border/X test relaxed")
            if m["touches_border"]:
                notes.append("touches image edge (tag cut off)")
            elif kind == "tag" and (m["ar"] > 1.25 or m["ring"] < 0.85):
                notes.append("tag partly hidden (visible part not square / border incomplete)")
            if m["side"] < 30:
                notes.append("very small")
            conf = "Low" if kind == "tag_blurred" else ("High" if not notes else "Medium")
            m.update(conf=conf, notes=notes)
            found.append(m)
        elif kind == "near_miss":
            near.append(m)
    found.sort(key=lambda d: -(d["ring"] + d["inner"]))
    kept = []
    for d in found:                                                             
        if all(iou(d["box"], k["box"]) < 0.3 for k in kept):
            kept.append(d)
    return kept, near


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="images")
    ap.add_argument("--out", default="output_detections")
    ap.add_argument("--csv", default="results.csv")
    ap.add_argument("--json", default="results.json")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows, js = [], {}
    colours = {"High": (0, 200, 0), "Medium": (0, 200, 255), "Low": (0, 128, 255)}      
    for path in sorted(glob.glob(os.path.join(args.images, "*.jpg")) + glob.glob(os.path.join(args.images, "*.png"))):
        name = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            print("cannot read", path); continue
        dets, near = detect(img)
        out = img.copy()
        for d in dets:
            x0, y0, x1, y1 = d["box"]
            cv2.rectangle(out, (x0, y0), (x1, y1), colours[d["conf"]], 2)
            cv2.putText(out, f"tag ({d['conf']})", (x0, max(y0 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colours[d["conf"]], 2)
        cv2.imwrite(os.path.join(args.out, name), out)
        boxes = [d["box"] for d in dets]
        notes = [n for d in dets for n in d["notes"]] + [f"near-miss candidate at {m['fill_box']}" for m in near]
        order = ["High", "Medium", "Low"]              
        if not dets:
            conf = "High" if not near else "Medium"      
        else:
            conf = max((d["conf"] for d in dets), key=order.index)
            if near:
                conf = max(conf, "Medium", key=order.index)
        self_check = "yes" if conf == "High" else "uncertain"   
        rows.append(dict(image_name=name, tags_found=len(dets),
                         bounding_boxes="; ".join(str(b) for b in boxes) if boxes else "none",
                         confidence=conf, i_think_correct=self_check, notes=" | ".join(dict.fromkeys(notes))))
        js[name] = dict(boxes=boxes, confidence=conf, notes=notes)
    if not rows:
        print("no images found in", args.images); return
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(args.json, "w") as f:
        json.dump(js, f, indent=1)
    for r in rows:
        print(f"{r['image_name']}  tags={r['tags_found']}  conf={r['confidence']:6s} {r['bounding_boxes']}  {r['notes']}")


if __name__ == "__main__":
    main()