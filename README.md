# CS566_Vision-Based-Dynamic-OBjects-Path-Prediction-for-Safer-Robot-Navigation-on-Construction-Sites
# Final Report – Vision-Based Dynamic Objects Path Prediction for Safer Robot Navigation on Construction Sites

**Authors:**  
- Liqun Xu – Department of Civil and Environmental Engineering, University of Wisconsin–Madison  
- Pakorn Boonpetch – Department of Materials Science and Engineering, University of Wisconsin–Madison  

**Contact:** liqun.xu@wisc.edu, pakorn.boonpetch@wisc.edu  

---

## 1. Overview

This project focuses on **predicting the future paths of dynamic objects (e.g., construction workers, equipment, and vehicles) from visual data** to improve the safety of robot navigation on active construction sites.

Our main goals are:

- Detect and track dynamic agents in construction scenes using visual sensing (e.g., stereo/RGB camera).
- Predict their short-horizon future trajectories (e.g., next 2–3 seconds).
- Visualize predictions in a way that is useful for a robot navigation stack (e.g., collision avoidance or risk-aware planning).
- Analyze failure cases and challenges specific to construction sites (occlusions, clutter, motion blur, etc.).

> _Replace this paragraph with a 3–5 sentence high-level summary tailored to your actual implementation and findings._

---

## 2. Motivation

> Paste / adapt content from `sec/2_motivation.tex` here.

Robots operating on construction sites face several unique challenges:

- **Highly dynamic environments:** workers, vehicles, and materials move unpredictably.
- **Unstructured layouts:** paths are not as clearly defined as in indoor or autonomous-driving settings.
- **Safety-critical interactions:** near-misses or collisions with workers and equipment are unacceptable.

Accurate **vision-based dynamic object path prediction** is therefore critical for:

- Anticipating potential collision scenarios.
- Giving the navigation system enough time to react.
- Supporting human–robot coexistence on busy job sites.

In this project, we target **short-term trajectory prediction** of dynamic agents using video data, as a stepping stone toward safer robot navigation.

---

## 3. Problem Definition

> Paste / adapt content from `sec/1_problemdef.tex` here.

We consider the following problem:

- **Input:** a sequence of past observations of dynamic objects in a construction scene, obtained from a moving camera (e.g., ZED 2i) mounted on a robot.  
  - Each object has:  
    - 2D/3D positions over the last few frames.  
    - Optionally: velocity, bounding box, or depth information.
- **Output:** predicted future positions of each tracked dynamic object over a fixed prediction horizon (e.g., next 90 frames).

Key questions:

- How can we robustly track and predict motion under noisy measurements, camera motion, and partial occlusion?
- How accurate and stable are these predictions in realistic construction footage?

---

## 4. Approach

> Here describe the **algorithmic pipeline** at a high level. You can summarize the approach you actually implemented.

Our pipeline consists of the following main components:

1. **Sensing & Input Data**
   - Stereo/RGB video captured on an active construction site (e.g., from a ZED 2i camera mounted on a quadruped robot).
   - Optional depth / point cloud information for projecting detections into 3D.

2. **Object Detection**
   - Use a pre-trained object detector (e.g., YOLO) to detect:
     - Workers / people  
     - Construction equipment (excavator, crane hook, etc.)  
     - Other relevant dynamic objects

3. **Multi-Object Tracking**
   - Associate detections across frames to form **tracks**:
     - Track IDs for each object  
     - Smoothed trajectories over time  
   - May use a combination of:
     - Motion model (e.g., Kalman filter)
     - Appearance / IoU-based association

4. **Trajectory Prediction**
   - For each tracked object, use its recent motion history to predict its future path.
   - Example prediction strategies (choose what you actually used):
     - Constant-velocity / constant-acceleration model.
     - Kalman filter time extrapolation.
     - Learning-based model (e.g., RNN/LSTM, transformer, etc.) if implemented.

5. **Visualization**
   - Overlay predicted trajectories on:
     - The original image view.
     - A bird’s-eye view (BEV) or world-coordinate view if available.
   - Use color-coded markers (e.g., color gradient along the timeline) to indicate time into the future.

> _Replace / refine the bullet points above so they match your actual implementation._

---

## 5. Implementation

> Here you describe practical details: environment, libraries, code structure, and how the system is run.

### 5.1. Software and Environment

- Programming language: **Python**
- Main libraries / frameworks:
  - `pyzed.sl` for ZED camera / SVO file reading
  - `OpenCV` for image processing and visualization
  - `NumPy` for numerical operations
  - `ultralytics` / YOLO model for object detection
- Hardware:
  - GPU: e.g., NVIDIA RTX 3090 Ti (for detection and potential deep models)
  - Camera: ZED 2i (video recorded as `.svo2` files)

### 5.2. Code Structure

> Tailor this section to your actual repository layout.

- `zed_traj_predict.py`  
  - Main script to:
    - Load SVO2 video  
    - Run detection + tracking  
    - Predict trajectories  
    - Visualize and optionally save results
- `utils/`
  - `transforms.py`: world–camera–image coordinate transformations.
  - `tracking.py`: track management and motion models.
  - `visualization.py`: rendering trajectories (e.g., colored spheres along predicted path).

Example command to run the pipeline:

```bash
python zed_traj_predict.py \
  --svo path/to/site.svo2 \
  --outdir runs/site01 \
  --prediction_horizon 90 \
  --fps 30 \
  --max_preview_fps 30
