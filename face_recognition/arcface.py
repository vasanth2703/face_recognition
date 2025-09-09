import os
import time
import cv2
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass

# Use InsightFace for ArcFace embeddings + detector
# (RetinaFace/SCRFD detector + glint360k ArcFace embeddings)
from insightface.app import FaceAnalysis


# ===============================
# Config
# ===============================
KNOWN_DB_DIR = "known_faces"
# Cosine similarity threshold; higher => stricter. Typical sweet spot 0.32~0.45
# Start with 0.35; increase if you get false positives, decrease if you miss.
MATCH_THRESHOLD = 0.35
# Desired capture device index
CAM_INDEX = 0
# Resize preview for speed (None to keep camera native size)
PREVIEW_WIDTH = 960  # set None to keep original


@dataclass
class PersonEmbedding:
    name: str
    embedding: np.ndarray  # L2-normalized vector


def l2_normalize(v: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    return v / max(np.linalg.norm(v), eps)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # Inputs should be L2-normalized
    return float(np.dot(a, b))


def prepare_insightface(ctx_id: int = -1) -> FaceAnalysis:
    """
    ctx_id: GPU id; -1 = CPU (safe default for Windows laptops without CUDA)
    """
    app = FaceAnalysis(name="buffalo_l")  # balanced model pack
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app


def image_files_in(folder: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(exts):
                paths.append(os.path.join(root, f))
    return paths


def load_known_database(app: FaceAnalysis, db_dir: str = KNOWN_DB_DIR) -> List[PersonEmbedding]:
    """
    Scans known_faces/PersonName/*.jpg and builds an average embedding per person.
    """
    if not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
        print(f"[DB] Created '{db_dir}'. Add person folders and images, then rerun.")
        return []

    # Map person -> list of embeddings
    person_embs: Dict[str, List[np.ndarray]] = {}

    # Expect structure known_faces/Name/xxx.jpg
    for name in sorted(os.listdir(db_dir)):
        person_dir = os.path.join(db_dir, name)
        if not os.path.isdir(person_dir):
            continue

        paths = image_files_in(person_dir)
        if not paths:
            print(f"[DB] No images for '{name}' in {person_dir}")
            continue

        per_images = 0
        for img_path in paths:
            img = cv2.imread(img_path)
            if img is None:
                print(f"[DB] Could not read image: {img_path}")
                continue

            # InsightFace expects BGR np array; app.get returns faces with embeddings
            faces = app.get(img)
            if len(faces) == 0:
                print(f"[DB] No face in: {img_path}")
                continue

            # Pick the largest face (more reliable)
            faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
            emb = faces[0].normed_embedding  # already L2-normalized
            if emb is None:
                print(f"[DB] No embedding for: {img_path}")
                continue

            person_embs.setdefault(name, []).append(emb.astype(np.float32))
            per_images += 1

        if per_images:
            print(f"[DB] Loaded {per_images} image(s) for '{name}'")

    # Average each person's embeddings to a single vector
    database: List[PersonEmbedding] = []
    for name, embs in person_embs.items():
        avg = l2_normalize(np.mean(np.stack(embs, axis=0), axis=0).astype(np.float32))
        database.append(PersonEmbedding(name=name, embedding=avg))

    print(f"[DB] Final database: {len(database)} person(s)")
    if not database:
        print(f"[DB] '{db_dir}' is empty. Add images like '{db_dir}/Mugil/1.jpg' and rerun.")
    return database


def annotate(frame: np.ndarray, bbox: Tuple[float, float, float, float],
             label: str, color=(0, 200, 0)) -> None:
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y2 - th - 10), (x1 + tw + 8, y2), color, -1)
    cv2.putText(frame, label, (x1 + 4, y2 - 4), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def identify(face_emb: np.ndarray, db: List[PersonEmbedding]) -> Tuple[str, float]:
    """
    Returns (name, similarity) if above threshold, else ("Unknown", best_sim).
    face_emb must be L2-normalized.
    """
    if not db:
        return "Unknown", 0.0
    sims = [cosine_similarity(face_emb, p.embedding) for p in db]
    best_idx = int(np.argmax(sims))
    best_sim = sims[best_idx]
    if best_sim >= MATCH_THRESHOLD:
        return db[best_idx].name, best_sim
    return "Unknown", best_sim


def resize_keep_aspect(image: np.ndarray, width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w == width:
        return image
    new_h = int(h * (width / w))
    return cv2.resize(image, (width, new_h), interpolation=cv2.INTER_LINEAR)


def live_recognition(app: FaceAnalysis, database: List[PersonEmbedding]) -> None:
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[Cam] Error: Could not open camera.")
        return

    print("[Live] Press 'q' to quit.")
    prev = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[Cam] Read failed.")
            break

        if PREVIEW_WIDTH:
            frame = resize_keep_aspect(frame, PREVIEW_WIDTH)

        t0 = time.time()
        faces = app.get(frame)  # returns list of faces with bbox, kps, det_score, normed_embedding
        for f in faces:
            emb = f.normed_embedding
            if emb is None:
                continue
            name, sim = identify(emb.astype(np.float32), database)
            label = f"{name} ({sim:.2f})" if name != "Unknown" else f"Unknown ({sim:.2f})"
            color = (0, 200, 0) if name != "Unknown" else (0, 0, 255)
            annotate(frame, f.bbox, label, color=color)

        # FPS
        dt = time.time() - t0
        fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 26), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Live Face Recognition (InsightFace ArcFace)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def recognize_in_image(app: FaceAnalysis, database: List[PersonEmbedding], image_path: str) -> None:
    if not os.path.exists(image_path):
        print(f"[Image] Not found: {image_path}")
        return
    img = cv2.imread(image_path)
    if img is None:
        print(f"[Image] Could not read: {image_path}")
        return

    faces = app.get(img)
    print(f"[Image] Found {len(faces)} face(s).")
    for f in faces:
        emb = f.normed_embedding
        if emb is None:
            continue
        name, sim = identify(emb.astype(np.float32), database)
        label = f"{name} ({sim:.2f})" if name != "Unknown" else f"Unknown ({sim:.2f})"
        color = (0, 200, 0) if name != "Unknown" else (0, 0, 255)
        annotate(img, f.bbox, label, color=color)
        print(f"  - {label}")

    cv2.imshow("Face Recognition Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    print("Initializing InsightFace (ArcFace) …")
    # Use CPU by default to avoid CUDA hassles on Windows
    app = prepare_insightface(ctx_id=-1)

    print("Loading known faces database …")
    database = load_known_database(app, KNOWN_DB_DIR)
    if not database:
        print("[DB] Empty database. Add images to 'known_faces/<Name>/' and rerun.")
        return

    print("\n=== Face Recognition ===")
    print("1) Live recognition from webcam")
    print("2) Recognize faces in a single image")
    print("3) Exit")
    choice = input("Enter choice (1/2/3): ").strip()

    if choice == "1":
        live_recognition(app, database)
    elif choice == "2":
        path = input("Enter image path: ").strip().strip('"')
        recognize_in_image(app, database, path)
    else:
        print("Bye.")


if __name__ == "__main__":
    main()
