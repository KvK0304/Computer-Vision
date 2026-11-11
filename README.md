# Visual Marker Tag Detector (Industrial Computer Vision)

A robust, classical OpenCV computer vision pipeline designed to detect industrial visual marker tags (orange square tag with a thick black border and a black 'X') across varying real-world conditions.

---

##  Method & Approach

Rather than relying purely on color thresholding (which fails when plain orange distractors are present) or deep learning, this detector uses a **4-stage physical evidence verification pipeline**:

1. **Preprocessing & Denoising:** Applies bilateral filtering to eliminate sensor noise/grain without smudging sharp edge boundaries.
2. **Illumination Equalization & Color Masking:** Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) on the $L$-channel in Lab color space to handle underexposure and dark color casts. Color candidate regions are extracted via an expanded HSV warm-color mask ($H \in [5, 35]$) to account for yellow-shifted lighting casts.
3. **Relative Contrast & Structural Verification:**
   * **Border Ratio (`BORDER_RATIO = 0.196`):** Measures expected black border thickness relative to the tag's inner square size (calibrated from `target_reference.png`).
   * **Relative Dark Thresholding:** Computes the percentage of the outer border ring (`ring_dark`) and inner fill (`inner_dark`) that are dark *relative* to the inner orange fill brightness ($0.5 \times \text{fill\_brightness}$). This allows the detector to succeed in dim lighting, bright glare, and heavy color casts.
4. **Blur & Edge Geometry Handling:** Measures Sobel edge gradients (`sharp`) to adaptively relax thresholds for blurry images while verifying central intersection darkness (`centre_dark`). Non-Maximum Suppression (NMS) merges overlapping candidate boxes.

---

## 📊 Results Summary (All 20 Images)

| Image Name | Tags Found | Bounding Boxes `[x_min, y_min, x_max, y_max]` | Confidence | Correct? | Notes / Condition Handled |
| :--- | :---: | :--- | :---: | :---: | :--- |
| `img_01.jpg` | 1 | `[220, 225, 307, 313]` | High | Yes | Baseline rotated tag |
| `img_02.jpg` | 1 | `[351, 67, 464, 179]` | High | Yes | Baseline tag + plain distractor shapes |
| `img_03.jpg` | 1 | `[199, 197, 228, 226]` | Medium | Yes | Small-scale tag embedded inside shape |
| `img_04.jpg` | 1 | `[205, 142, 396, 332]` | High | Yes | Large-scale tag, slight tilt |
| `img_05.jpg` | 1 | `[351, 114, 467, 230]` | High | Yes | Tag adjacent to orange square distractor |
| `img_06.jpg` | 1 | `[103, 184, 237, 318]` | High | Yes | Heavy orientation/rotation (~30°) |
| `img_07.jpg` | 1 | `[314, 192, 389, 267]` | High | Yes | Severely underexposed / dark green cast |
| `img_08.jpg` | 1 | `[151, 67, 222, 138]` | High | Yes | Bright pink background / yellow-shifted cast |
| `img_09.jpg` | 1 | `[284, 155, 342, 213]` | Low | Yes | Heavy Gaussian / motion blur |
| `img_10.jpg` | 1 | `[60, 342, 125, 407]` | High | Yes | Corner placement with distractors |
| `img_11.jpg` | 1 | `[116, 269, 224, 377]` | High | Yes | High sensor noise / digital grain overlay |
| `img_12.jpg` | 1 | `[404, 65, 461, 166]` | Medium | Yes | ~50% partial occlusion by gray box |
| `img_13.jpg` | 1 | `[410, 230, 488, 307]` | High | Yes | Highly cluttered scene with orange distractor |
| `img_14.jpg` | 1 | `[385, 242, 442, 303]` | High | Yes | Olive lighting cast + distractors |
| `img_15.jpg` | 2 | `[360, 107, 424, 170]; [257, 255, 335, 332]` | High | Yes | Multi-tag image (2 tags present) |
| `img_16.jpg` | 2 | `[99, 78, 130, 109]; [524, 246, 555, 277]` | Medium | Yes | Multi-tag image (2 ultra-small tags ~31x31 px) |
| `img_17.jpg` | 0 | `none` | High | Yes | Negative test image (plain orange distractor bar) |
| `img_18.jpg` | 0 | `none` | High | Yes | Negative test image (plain orange distractors) |
| `img_19.jpg` | 0 | `none` | High | Yes | Negative test image (salmon/orange distractor circle) |
| `img_20.jpg` | 1 | `[545, 190, 640, 298]` | Medium | Yes | Partial image boundary crop at right edge |

---

##  Failure & Robustness Analysis

* **Illumination Shifts & Color Casts (`img_07.jpg`, `img_08.jpg`):** Fixed color thresholds fail when underexposure drops lightness or lighting shifts orange to yellow. By using CLAHE on the $L$-channel and relative contrast ratios ($g_{\text{border}} < 0.5 \times g_{\text{fill}}$), detection remains stable across lighting extremes.
* **Heavy Blur (`img_09.jpg`):** Motion/Gaussian blur smudges the sharp 'X' into the orange fill, reducing strict edge scores. The pipeline profiles Sobel edge gradients (`sharp < 0.6`) and safely relaxes the ring test while confirming central dark crossing points (`centre_dark`).
* **Partial Occlusions & Boundary Crops (`img_12.jpg`, `img_20.jpg`):** Covering part of the tag alters bounding box aspect ratios. The algorithm detects image edge contacts (`touches_border`) and evaluates structural hulls to preserve bounding box integrity.
* **False Positive Distractors (`img_17.jpg`–`img_19.jpg`):** Plain orange shapes pass color thresholding. The structural ratio tests (`ring_dark >= 0.65` and `inner_dark >= 0.10`) cleanly reject plain shapes, yielding 0 false positives.

---

##  How to Run

### 1. Prerequisites & Dependencies
Ensure Python 3 is installed along with `opencv-python` and `numpy`:
```bash
python3 -m pip install opencv-python numpy
