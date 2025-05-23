# constellation_utils.py

import numpy as np
import json
from numba import njit
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree

# === 1) Star detection via threshold + DBSCAN on pixel coords ===
def get_star_coords(
    img_gray: np.ndarray,
    star_thresh: int = 200,
    eps: float = 8,
    min_samples: int = 5,
    max_points: int = 5000
) -> np.ndarray:
    """
    img_gray: 2D array of grayscale pixel values
    Returns: Nx2 array of (x,y) centroids of detected stars
    """
    ys, xs = np.where(img_gray > star_thresh)
    if xs.size == 0:
        return np.empty((0,2), dtype=np.float64)

    pts = np.column_stack((xs, ys))
    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
    stars = []
    for lab in set(labels):
        if lab < 0:
            continue
        cluster = pts[labels == lab]
        stars.append(cluster.mean(axis=0))
    return np.array(stars, dtype=np.float64)


# === 2) Normalize coords (Numba for speed) ===
@njit
def normalize_coords(points: np.ndarray) -> np.ndarray:
    n = points.shape[0]
    if n == 0:
        return points
    cx = 0.0; cy = 0.0
    for i in range(n):
        cx += points[i,0]
        cy += points[i,1]
    cx /= n; cy /= n

    norm = 0.0
    for i in range(n):
        dx = points[i,0] - cx
        dy = points[i,1] - cy
        norm += dx*dx + dy*dy
    norm = norm**0.5

    out = np.empty_like(points)
    for i in range(n):
        out[i,0] = points[i,0] - cx
        out[i,1] = points[i,1] - cy
    if norm > 0.0:
        for i in range(n):
            out[i] /= norm
    return out


# === 3) Load & reduce constellations with dynamic eps_model ===
def load_constellations(path: str = "constellations.lines.json") -> list:
    """
    Returns list of dicts:
      [{'name': str,
        'arr': original points array,
        'norm_pts': reduced & normalized model points,
        'line_segs': list of line segments for plotting}, ...]
    """
    data = json.load(open(path, encoding='utf-8'))
    consts = []
    for feat in data["features"]:
        name = feat["id"]
        raw_pts = []
        segs = []
        for seg in feat["geometry"]["coordinates"]:
            seg = np.array(seg, dtype=np.float64)
            raw_pts += [(ra % 360, dec) for ra, dec in seg]
            segs.append(seg)
        # dedupe
        seen = set(); pts = []
        for p in raw_pts:
            if p not in seen:
                seen.add(p); pts.append(p)
        if len(pts) < 4:
            continue

        arr = np.array(pts)
        norm_pts = normalize_coords(arr)
        eps_model = 0.1 / np.sqrt(len(norm_pts))
        labels = DBSCAN(eps=eps_model, min_samples=1).fit_predict(norm_pts)
        reduced = np.array([norm_pts[labels==lab].mean(axis=0) for lab in set(labels)])
        reduced = normalize_coords(reduced)
        consts.append({
            'name': name,
            'arr': arr,
            'norm_pts': reduced,
            'line_segs': segs
        })
    return consts


# === 4) Similarity estimation & RANSAC matching ===
def estimate_similarity(a: np.ndarray, b: np.ndarray):
    """
    Procrustes similarity: scale s, rotation R, translation t
    that best maps a→b
    """
    ca = a.mean(0); cb = b.mean(0)
    A = a - ca; B = b - cb
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1] *= -1
        R = Vt.T @ U.T
    s = np.trace(R @ H) / (A**2).sum()
    t = cb - s*(R @ ca)
    return s, R, t


def ransac_match(
    det_norm: np.ndarray,
    mod_norm: np.ndarray,
    sample_k: int,
    iters: int = 100,
    thresh: float = 0.05,
    min_ratio: float = 0.8
):
    """
    RANSAC to align model→detected. Returns best_res or None.
    best_res = {'s','R','t','tm','dists','idxs','inliers','score'}
    """
    n, k = det_norm.shape[0], mod_norm.shape[0]
    if n < sample_k or k < sample_k:
        return None

    tree = cKDTree(det_norm)
    best_score = np.inf
    best_res = None

    for _ in range(iters):
        mi = np.random.choice(k, sample_k, replace=False)
        di = np.random.choice(n, sample_k, replace=False)
        try:
            s, R, t = estimate_similarity(mod_norm[mi], det_norm[di])
        except:
            continue

        tm = (mod_norm @ R.T) * s + t
        dists, idxs = tree.query(tm)
        inliers = dists < thresh
        if inliers.sum() < min_ratio * k:
            continue

        err = dists[inliers].mean()
        score = err + abs(1 - s)*0.5
        if score < best_score:
            best_score = score
            best_res = {
                's': s, 'R': R, 't': t,
                'tm': tm, 'dists': dists,
                'idxs': idxs, 'inliers': inliers,
                'score': score
            }

    return best_res


# === 5) find_top3 sorted by ratio & score ===
def find_top3(
    coords: np.ndarray,
    consts: list,
    user_k: int = None
) -> list:
    """
    coords: Nx2 detected star centroids
    consts: output of load_constellations()
    user_k: if provided, only compare against constellations of that many model points
    Returns top‑3 list of tuples:
       (ratio, score, model_k, const_dict, ransac_out)
    """
    det_norm = normalize_coords(coords)
    candidates = []

    for c in consts:
        km = c['norm_pts'].shape[0]
        if user_k and km != user_k:
            continue
        out = ransac_match(det_norm, c['norm_pts'], sample_k=(user_k or 4))
        if not out:
            continue
        ratio = out['inliers'].sum() / km
        candidates.append((ratio, out['score'], km, c, out))

    # sort: highest ratio, lowest score, largest model size
    candidates.sort(key=lambda x: (-x[0], x[1], -x[2]))
    return candidates[:3]


# === 6) Orchestrator: from image → top3 match & accuracy ===
def detect_and_match(
    img_gray: np.ndarray,
    star_thresh: int = 200,
    eps: float = 8,
    min_samples: int = 5,
    max_points: int = 5000,
    user_k: int = None,
    const_path: str = "constellations.lines.json"
):
    """
    Full pipeline:
      1. detect stars from img_gray with threshold
      2. load and reduce constellations
      3. run find_top3 with optional user_k
    Returns:
      coords: detected star centroids (array Nx2)
      top3: list of (ratio, score, k, const_dict, out)
    """
    coords = get_star_coords(img_gray, star_thresh, eps, min_samples, max_points)
    consts = load_constellations(const_path)
    top3 = find_top3(coords, consts, user_k)
    return coords, top3
