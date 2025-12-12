import sys
import cv2
import numpy as np
import pyzed.sl as sl
import math
import time
from collections import deque
from ultralytics import YOLO

# --- UTILITY: Coordinate Transforms ---

def to_world(cam_point, cam_pose_mat):
    pt_cam_h = np.array([cam_point[0], cam_point[1], cam_point[2], 1.0])
    pt_world_h = np.dot(cam_pose_mat, pt_cam_h)
    return pt_world_h[:3]

def to_screen(world_point, cam_pose_mat, cam_matrix):
    world_to_cam = np.linalg.inv(cam_pose_mat)
    pt_world_h = np.array([world_point[0], world_point[1], world_point[2], 1.0])
    pt_cam_h = np.dot(world_to_cam, pt_world_h)
    pt_cam = pt_cam_h[:3]
    if pt_cam[2] <= 0: return None 
    fx, fy = cam_matrix[0, 0], cam_matrix[1, 1]
    cx, cy = cam_matrix[0, 2], cam_matrix[1, 2]
    u = int((pt_cam[0] * fx / pt_cam[2]) + cx)
    v = int((pt_cam[1] * fy / pt_cam[2]) + cy)
    return (u, v)

def xywh2abcd(xywh, h_img, w_img):
    output = np.zeros((4, 2))
    x_c, y_c, w, h = xywh
    output[0] = [x_c - w/2, y_c - h/2]
    output[1] = [x_c + w/2, y_c - h/2]
    output[2] = [x_c + w/2, y_c + h/2]
    output[3] = [x_c - w/2, y_c + h/2]
    return output

# --- VISUALIZATION HELPERS ---

def overlay_image(background, foreground, location):
    bx, by = location
    h, w = foreground.shape[:2]
    if bx < 0 or by < 0 or bx + w > background.shape[1] or by + h > background.shape[0]:
        return
    if foreground.shape[2] == 4:
        alpha_mask = foreground[:, :, 3] / 255.0
        img_rgb = foreground[:, :, :3]
        for c in range(0, 3):
            background[by:by+h, bx:bx+w, c] = (alpha_mask * img_rgb[:, :, c] + 
                                               (1.0 - alpha_mask) * background[by:by+h, bx:bx+w, c])
    else:
        background[by:by+h, bx:bx+w] = foreground

def draw_triangle(image, center, direction_vector, size=12, color=(0, 255, 255)):
    cx, cy = center
    length = math.sqrt(direction_vector[0]**2 + direction_vector[1]**2)
    if length < 1e-3: return 
    ux, uy = direction_vector[0] / length, direction_vector[1] / length
    px, py = -uy, ux
    tip = (int(cx + ux * size), int(cy + uy * size))
    base_left = (int(cx - ux * size * 0.5 + px * size * 0.5), int(cy - uy * size * 0.5 + py * size * 0.5))
    base_right = (int(cx - ux * size * 0.5 - px * size * 0.5), int(cy - uy * size * 0.5 - py * size * 0.5))
    pts = np.array([tip, base_left, base_right], np.int32)
    cv2.fillPoly(image, [pts], color)
    cv2.polylines(image, [pts], True, (0,0,0), 1)

def draw_gradient_line(image, start_pt, end_pt, steps=10):
    if start_pt is None or end_pt is None: return
    for i in range(steps):
        t1 = i / steps
        t2 = (i + 1) / steps
        x1 = int(start_pt[0] + (end_pt[0] - start_pt[0]) * t1)
        y1 = int(start_pt[1] + (end_pt[1] - start_pt[1]) * t1)
        x2 = int(start_pt[0] + (end_pt[0] - start_pt[0]) * t2)
        y2 = int(start_pt[1] + (end_pt[1] - start_pt[1]) * t2)
        g_val = int(255 * t1)
        color = (0, g_val, 255) 
        cv2.line(image, (x1, y1), (x2, y2), color, 2)

def draw_bev_map(bev_img, robot_pos, robot_pose_mat, objects_data, person_icon, machine_icon, scale=15, map_size=500):
    bev_img.fill(245) 
    cx, cy = map_size // 2, map_size // 2
    
    def to_map(wx, wz):
        rel_x = wx - robot_pos[0]
        rel_z = wz - robot_pos[2]
        u = int(cx + (rel_x * scale))
        v = int(cy - (rel_z * scale))
        return (u, v)

    # Grid
    grid_spacing_m = 1.0
    off_x = int((robot_pos[0] % grid_spacing_m) * scale)
    off_z = int((robot_pos[2] % grid_spacing_m) * scale)
    for i in range(-map_size, map_size, int(grid_spacing_m * scale)):
        u = cx + i - off_x
        if 0 <= u < map_size: cv2.line(bev_img, (u, 0), (u, map_size), (220, 220, 220), 1)
        v = cy + i + off_z
        if 0 <= v < map_size: cv2.line(bev_img, (0, v), (map_size, v), (220, 220, 220), 1)

    # Objects
    for obj in objects_data:
        pos = obj['pos']
        vel = obj['vel']
        label = obj['label']
        dims = obj['dims'] 
        is_moving = obj['is_moving'] # New flag
        
        u, v = to_map(pos[0], pos[2])
        
        if 0 <= u < map_size and 0 <= v < map_size:
            # Color: If not moving, draw in GRAY (Ghost mode) to show we detected it but ignored it
            color = (0, 0, 255) if label != 0 else (0, 180, 0)
            if not is_moving:
                color = (180, 180, 180) # Gray for static objects

            # Bounding Box
            w2 = dims[0] / 2.0 
            l2 = dims[2] / 2.0 
            corners_world = [(pos[0]-w2, pos[2]-l2), (pos[0]+w2, pos[2]-l2),
                             (pos[0]+w2, pos[2]+l2), (pos[0]-w2, pos[2]+l2)]
            corners_map = [to_map(cw[0], cw[1]) for cw in corners_world]
            corners_map_np = np.array(corners_map, np.int32).reshape((-1, 1, 2))
            
            overlay = bev_img.copy()
            cv2.fillPoly(overlay, [corners_map_np], color)
            cv2.addWeighted(overlay, 0.3, bev_img, 0.7, 0, bev_img)
            cv2.polylines(bev_img, [corners_map_np], True, color, 1)

            # Icon
            icon = None
            if label == 0: icon = person_icon
            else: icon = machine_icon
            
            if icon is not None:
                h_icon, w_icon = icon.shape[:2]
                overlay_image(bev_img, icon, (u - w_icon // 2, v - h_icon // 2))

            # Gradient Trajectory (Only if confirmed moving)
            if is_moving:
                pred_x = pos[0] + (vel[0] * 2.0)
                pred_z = pos[2] + (vel[2] * 2.0)
                pu, pv = to_map(pred_x, pred_z)
                draw_gradient_line(bev_img, (u, v), (pu, pv), steps=10)

    # Robot Axis
    axis_len = 25 
    vec_x = robot_pose_mat[0:3, 0] 
    end_x = int(cx + (vec_x[0] * axis_len))
    end_z_x = int(cy - (vec_x[2] * axis_len))
    cv2.line(bev_img, (cx, cy), (end_x, end_z_x), (0, 0, 255), 2)
    vec_z = robot_pose_mat[0:3, 2]
    end_x_z = int(cx + (vec_z[0] * axis_len))
    end_z_z = int(cy - (vec_z[2] * axis_len))
    cv2.line(bev_img, (cx, cy), (end_x_z, end_z_z), (255, 0, 0), 2)
    cv2.circle(bev_img, (cx, cy), 4, (0, 255, 0), -1)

# --- ROBUST KALMAN FILTER ---

class RobustKalmanFilter:
    def __init__(self, initial_pos, dt=0.033):
        self.kf = cv2.KalmanFilter(6, 3)
        self.dt = dt
        self.kf.transitionMatrix = np.eye(6, dtype=np.float32)
        for i in range(3): self.kf.transitionMatrix[i, i+3] = dt
        self.kf.measurementMatrix = np.eye(3, 6, dtype=np.float32)
        
        # Lower process noise to assume smoother/slower motion
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.005 
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.1
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)
        
        self.kf.statePost = np.array([[initial_pos[0]], [initial_pos[1]], [initial_pos[2]], [0], [0], [0]], dtype=np.float32)
        self.last_update_time = time.time()
        
        # --- ROBUSTNESS BUFFER ---
        # We store the last 30 positions (approx 1 second)
        self.pos_history = deque(maxlen=30)
        self.pos_history.append(initial_pos)
        self.is_moving = False

    def predict(self):
        return self.kf.predict()

    def update(self, measurement, current_time):
        dt = current_time - self.last_update_time
        if dt > 0:
            for i in range(3): self.kf.transitionMatrix[i, i+3] = dt
        self.last_update_time = current_time
        
        self.kf.correct(np.array([[measurement[0]], [measurement[1]], [measurement[2]]], dtype=np.float32))
        
        curr_pos = self.kf.statePost[:3].flatten()
        curr_vel = self.kf.statePost[3:].flatten()
        
        # --- THE FIX: NET DISPLACEMENT CHECK ---
        self.pos_history.append(curr_pos)
        
        # Calculate distance between current position and position 1 second ago (or max history)
        start_pos = self.pos_history[0]
        net_dist = np.linalg.norm(curr_pos - start_pos)
        
        # THRESHOLD: Object must have moved 0.5 METERS total to be considered moving.
        # This filters out vibration/jitter (which usually stays within 10-20cm).
        if net_dist > 0.5:
            self.is_moving = True
        else:
            self.is_moving = False
            # Force velocity to zero to stop "creeping" lines
            curr_vel = np.array([0.0, 0.0, 0.0])
            
        return curr_pos, curr_vel, self.is_moving

class TrackerManager:
    def __init__(self):
        self.filters = {}
    def process(self, obj_id, current_world_pos, current_time):
        if obj_id not in self.filters:
            self.filters[obj_id] = RobustKalmanFilter(current_world_pos)
            return current_world_pos, np.array([0,0,0]), False
            
        kf = self.filters[obj_id]
        kf.predict()
        return kf.update(current_world_pos, current_time)

# --- MAIN ---

def main():
    SVO_PATH = "data/site.svo2"  # <--- UPDATE THIS
    OUTPUT_FILE = "robust_tracking.mp4"
    
    print("Loading YOLO...")
    model = YOLO("yolov8n.pt") 

    print("Loading Icons...")
    person_icon_raw = cv2.imread("person_icon.png", cv2.IMREAD_UNCHANGED)
    machine_icon_raw = cv2.imread("machine_icon.png", cv2.IMREAD_UNCHANGED)
    ICON_SIZE = 24
    person_icon = cv2.resize(person_icon_raw, (ICON_SIZE, ICON_SIZE), interpolation=cv2.INTER_AREA) if person_icon_raw is not None else None
    machine_icon = cv2.resize(machine_icon_raw, (ICON_SIZE, ICON_SIZE), interpolation=cv2.INTER_AREA) if machine_icon_raw is not None else None

    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(SVO_PATH)
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL 

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Error opening SVO.")
        sys.exit()

    tracking_params = sl.PositionalTrackingParameters()
    tracking_params.set_as_static = False
    zed.enable_positional_tracking(tracking_params)

    obj_param = sl.ObjectDetectionParameters()
    obj_param.enable_tracking = True
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    zed.enable_object_detection(obj_param)
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    
    objects = sl.Objects()
    image_left = sl.Mat()
    cam_pose = sl.Pose()
    tmp_transform = sl.Transform()
    
    cam_info = zed.get_camera_information()
    left_cam = cam_info.camera_configuration.calibration_parameters.left_cam
    cam_matrix = np.array([[left_cam.fx, 0, left_cam.cx], [0, left_cam.fy, left_cam.cy], [0, 0, 1]])
    w, h = cam_info.camera_configuration.resolution.width, cam_info.camera_configuration.resolution.height

    BEV_SIZE = 500
    final_w = w + BEV_SIZE
    final_h = max(h, BEV_SIZE)
    out = cv2.VideoWriter(OUTPUT_FILE, cv2.VideoWriter_fourcc(*'mp4v'), 30, (final_w, final_h))

    tracker_manager = TrackerManager()
    print(f"Processing... Saving to {OUTPUT_FILE}")

    while zed.grab() == sl.ERROR_CODE.SUCCESS:
        current_time = time.time()
        zed.retrieve_image(image_left, sl.VIEW.LEFT)
        frame_cv = image_left.get_data()[:,:,:3]
        frame_cv = np.ascontiguousarray(frame_cv)

        state = zed.get_position(cam_pose, sl.REFERENCE_FRAME.WORLD)
        if state == sl.POSITIONAL_TRACKING_STATE.OK:
            cam_pose.pose_data(tmp_transform)
            cam_pose_mat = tmp_transform.m 
            robot_pos = cam_pose.get_translation().get()
        else:
            cam_pose_mat = np.eye(4)
            robot_pos = [0,0,0]

        results = model.predict(frame_cv, verbose=False, conf=0.20, iou=0.45, agnostic_nms=True)
        detections = []
        for det in results[0].boxes:
            tmp_obj = sl.CustomBoxObjectData()
            tmp_obj.bounding_box_2d = xywh2abcd(det.xywh[0].cpu().numpy(), h, w)
            tmp_obj.label = int(det.cls[0])
            tmp_obj.probability = float(det.conf[0])
            tmp_obj.is_grounded = False 
            detections.append(tmp_obj)

        zed.ingest_custom_box_objects(detections)
        zed.retrieve_objects(objects, obj_runtime_param)

        objects_for_bev = []

        for obj in objects.object_list:
            raw_pos = obj.position 
            if np.isnan(raw_pos[0]): continue

            pos_world_noisy = to_world(raw_pos, cam_pose_mat) 
            
            # --- PROCESS WITH ROBUST FILTER ---
            smooth_pos, smooth_vel, is_moving = tracker_manager.process(obj.id, pos_world_noisy, current_time)
            
            # Only calculate speed if the filter confirms actual movement > 0.5m
            speed = np.linalg.norm(smooth_vel) if is_moving else 0.0

            objects_for_bev.append({
                'pos': smooth_pos, 
                'vel': smooth_vel, 
                'label': obj.label,
                'dims': obj.dimensions,
                'is_moving': is_moving
            })

            # Visualize on Camera View (ONLY if confirmed moving)
            if is_moving:
                pred_world = smooth_pos + (smooth_vel * 2.0)
                pt_current = to_screen(smooth_pos, cam_pose_mat, cam_matrix)
                pt_future = to_screen(pred_world, cam_pose_mat, cam_matrix)
                
                if pt_current and pt_future:
                    draw_gradient_line(frame_cv, pt_current, pt_future, steps=10)
                    screen_dir = (pt_future[0] - pt_current[0], pt_future[1] - pt_current[1])
                    draw_triangle(frame_cv, pt_current, screen_dir, size=14, color=(0, 255, 255))
                    label = f"{speed:.2f} m/s"
                    cv2.putText(frame_cv, label, (pt_current[0]+15, pt_current[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        bev_img = np.zeros((BEV_SIZE, BEV_SIZE, 3), dtype=np.uint8)
        draw_bev_map(bev_img, robot_pos, cam_pose_mat, objects_for_bev, person_icon, machine_icon, scale=15, map_size=BEV_SIZE)

        combined = np.zeros((final_h, final_w, 3), dtype=np.uint8)
        combined[:h, :w] = frame_cv
        combined[:BEV_SIZE, w:] = bev_img
        
        out.write(combined)
        cv2.imshow("Robust BEV Tracker", combined)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    out.release()
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()