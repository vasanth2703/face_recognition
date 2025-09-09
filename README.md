# Project Report: Face Recognition using ArcFace

## 1. Introduction

This project implements a real-time face recognition system leveraging the ArcFace embedding method via the InsightFace library. The primary goal is to identify known individuals from a pre-built database, either through a live camera feed or from static image files. This system is designed for applications requiring robust and accurate facial identification.

## 2. Features

The system provides the following key functionalities:

*   **Known Face Database Management**: Automatically scans a specified directory (`known_faces/PersonName/`) to build a database of known individuals. It computes an average L2-normalized embedding for each person from multiple sample images.
*   **Live Face Recognition**: Utilizes a connected camera to detect faces in real-time, extract their embeddings, and compare them against the known face database to identify individuals.
*   **Static Image Face Recognition**: Processes a given image file to detect faces and identify them against the known face database.
*   **Visual Annotations**: Displays bounding boxes, names, and similarity scores on detected faces in both live and static image modes.

## 3. Technical Details

### 3.1. Core Components

*   **InsightFace Library**: The project heavily relies on the `insightface` library, specifically the `FaceAnalysis` app with the "buffalo_l" model pack. This provides the underlying face detection (RetinaFace/SCRFD) and ArcFace embedding generation (glint360k).
*   **ArcFace Embeddings**: ArcFace is a state-of-the-art face recognition method that produces highly discriminative deep features (embeddings) for faces. These embeddings are L2-normalized vectors.
*   **Cosine Similarity**: Used to measure the similarity between a detected face's embedding and the embeddings in the known face database. A higher cosine similarity indicates a closer match.

### 3.2. Key Functions

*   `l2_normalize(v: np.ndarray)`: Normalizes a vector to have a unit L2 norm. Essential for consistent similarity comparisons.
*   `cosine_similarity(a: np.ndarray, b: np.ndarray)`: Calculates the cosine similarity between two L2-normalized vectors.
*   `prepare_insightface(ctx_id: int = -1)`: Initializes the InsightFace `FaceAnalysis` app. `ctx_id` can be set to a GPU ID for acceleration, or -1 for CPU-only processing.
*   `image_files_in(folder: str)`: Helper function to recursively find image files within a given directory.
*   `load_known_database(app: FaceAnalysis, db_dir: str = KNOWN_DB_DIR)`: Scans the `db_dir` (default `known_faces`) for subdirectories named after individuals. For each person, it processes all images, extracts face embeddings, and computes an average embedding for that person. This average embedding is then stored in the database.
*   `annotate(frame: np.ndarray, bbox: Tuple, label: str, color: Tuple)`: Draws a bounding box and a label (name and similarity score) on a given image frame.
*   `identify(face_emb: np.ndarray, db: List[PersonEmbedding])`: Compares a given face embedding against all known embeddings in the database. It returns the name of the best match and its similarity score if the score exceeds `MATCH_THRESHOLD`, otherwise it returns "Unknown".
*   `resize_keep_aspect(image: np.ndarray, width: int)`: Resizes an image to a specified width while maintaining its aspect ratio.
*   `live_recognition(app: FaceAnalysis, database: List[PersonEmbedding])`: Captures video from the specified camera (`CAM_INDEX`), performs real-time face detection and recognition, and displays the annotated frames. Press 'q' to quit.
*   `recognize_in_image(app: FaceAnalysis, database: List[PersonEmbedding], image_path: str)`: Loads a static image from `image_path`, performs face detection and recognition, and displays the annotated image.

### 3.3. Configuration Parameters

*   `KNOWN_DB_DIR`: Directory where known faces are stored (e.g., `known_faces`).
*   `MATCH_THRESHOLD`: Cosine similarity threshold for identifying a known face. A value of 0.35 is a good starting point.
*   `CAM_INDEX`: Index of the camera to use for live recognition (e.g., 0 for default camera).
*   `PREVIEW_WIDTH`: Desired width for the live camera preview. Set to `None` to use the camera's native resolution.

## 4. Setup and Usage

1.  **Install Dependencies**: Ensure you have `opencv-python`, `numpy`, and `insightface` installed. You might need to install `onnxruntime` or `onnxruntime-gpu` depending on your `ctx_id` setting.
2.  **Prepare Known Faces**: Create subdirectories within the `known_faces` folder, with each subdirectory named after a person (e.g., `known_faces/John_Doe/`). Place multiple clear images of that person inside their respective folder.
3.  **Run the Script**: Execute `arcface.py`. The script will load the known faces database and then initiate live face recognition using your webcam. You can also modify the script to call `recognize_in_image` for static image analysis.

## 5. Conclusion and Future Work

This project provides a functional foundation for face recognition. Future enhancements could include:

*   **Performance Optimization**: Further optimizing the inference speed for higher frame rates.
*   **User Interface**: Developing a more interactive graphical user interface for easier database management and recognition.
*   **Robustness**: Improving robustness against varying lighting conditions, poses, and occlusions.
*   **Logging**: Implementing detailed logging for recognition events and system performance.
