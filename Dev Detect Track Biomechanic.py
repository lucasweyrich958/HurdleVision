import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from scipy.signal import find_peaks, savgol_filter
import re
import os

# --- 1. MODEL & CONFIGURATION SETUP ---
ATHLETE_CLASS_NAME = 'athlete'
HURDLE_CLASS_NAME = 'hurdle'
FINISH_LINE_CLASS_NAME = 'finish' 

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    detection_model_path = os.path.join(BASE_DIR, "models", "yolo11m_tuned.pt")
    detection_model = YOLO(detection_model_path)
    names_to_ids = {name: i for i, name in detection_model.names.items()}
    
    athlete_class_id = names_to_ids[ATHLETE_CLASS_NAME]
    hurdle_class_id = names_to_ids[HURDLE_CLASS_NAME]
    finish_line_class_id = names_to_ids[FINISH_LINE_CLASS_NAME]
    
except Exception as e:
    print(f"--- ERROR ---")
    print(f"Could not load models or find all class IDs. Please check the class names at the top of the script.")
    print(f"Actual names found in model: {detection_model.names if 'detection_model' in locals() else 'Could not load model'}")
    print(f"Original error: {e}")
    exit()

# --- Configuration ---
START_MOTION_THRESHOLD = 20
DYNAMIC_CORRIDOR_WIDTH = 150
POSE_WINDOW_FRAMES = 20 

# --- Global variables for athlete selection ---
first_frame_athletes, target_athlete_id, selection_done, click_point = {}, None, False, None

def select_athlete(event, x, y, flags, param):
    global click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (x, y)

def pass_one_track_athlete(video_path, detection_model, target_athlete_id):
    print("\n--- Starting Pass 1: Tracking athlete's bounding box... ---")
    cap = cv2.VideoCapture(video_path)
    
    trajectory_log = []
    frame_number = 0
    current_state = "PRE_RACE"
    race_start_frame = 0
    starting_block_center = None
    last_known_box = None

    for track_id, box in first_frame_athletes.items():
        if track_id == target_athlete_id:
            starting_block_center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            last_known_box = box
            break

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_number += 1
        
        results = detection_model.track(frame, persist=True, verbose=False, tracker='botsort.yaml')
        target_box = None
        
        all_athletes_this_frame = []
        if results and results[0].boxes.id is not None:
            for box, cls_id in zip(results[0].boxes.xyxy.cpu(), results[0].boxes.cls.cpu()):
                if int(cls_id) == athlete_class_id:
                    all_athletes_this_frame.append(box.numpy().astype(int))
        
        if current_state == "PRE_RACE":
            if results and results[0].boxes.id is not None:
                track_ids_list = results[0].boxes.id.cpu().tolist()
                if target_athlete_id in track_ids_list:
                    target_index = track_ids_list.index(target_athlete_id)
                    box = results[0].boxes.xyxy.cpu()[target_index].numpy().astype(int)
                    last_known_box = box
                    athlete_center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
                    if race_start_frame == 0 and np.linalg.norm(athlete_center - starting_block_center) > START_MOTION_THRESHOLD:
                        current_state = "RACE_ANALYSIS"
                        race_start_frame = frame_number
                        print(f"🚀 Motion detected! Race starts at frame: {race_start_frame}")
        
        elif current_state == "RACE_ANALYSIS":
            if last_known_box is not None:
                last_center_x = (last_known_box[0] + last_known_box[2]) / 2
                half_width = DYNAMIC_CORRIDOR_WIDTH / 2
                tracking_corridor = (last_center_x - half_width, last_center_x + half_width)
                candidates_in_corridor = [b for b in all_athletes_this_frame if tracking_corridor[0] < ((b[0] + b[2]) / 2) < tracking_corridor[1]]
                
                if candidates_in_corridor:
                    last_center = np.array([(last_known_box[0] + last_known_box[2]) / 2, (last_known_box[1] + last_known_box[3]) / 2])
                    best_candidate = min(candidates_in_corridor, key=lambda b: np.linalg.norm(np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]) - last_center))
                    target_box = best_candidate
                else:
                    target_box = last_known_box

        if current_state == "RACE_ANALYSIS" and target_box is not None:
            last_known_box = target_box
            box_str = np.array2string(target_box)
            trajectory_log.append({'frame': frame_number, 'box': box_str})

    cap.release()
    if trajectory_log: pd.DataFrame(trajectory_log).to_csv("pass_1_trajectory_log.csv", index=False)
    print(f"✅ Pass 1 Complete. Logged {len(trajectory_log)} data points.")
    return race_start_frame

def analyze_trajectory_for_peaks(trajectory_log_path):
    print("\n--- Analyzing Trajectory: Using Enhanced Two-Factor Peak Detection... ---")
    try:
        df = pd.read_csv(trajectory_log_path)
    except FileNotFoundError: return []
    if df.empty: return []

    box_coords = df['box'].str.extract(r'\[\s*(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s*\]').astype(float)
    df['aspect_ratio'] = (box_coords[2] - box_coords[0]) / (box_coords[3] - box_coords[1])
    df['y_center'] = (box_coords[1] + box_coords[3]) / 2

    inverted_y = -df['y_center']
    window_length = min(21, len(df) - 1 if len(df) % 2 == 0 else len(df))
    if window_length < 5: return []
    df['y_smooth'] = savgol_filter(inverted_y, window_length, 3) 
    df['ar_smooth'] = savgol_filter(df['aspect_ratio'], window_length, 3)

    y_peaks, y_props = find_peaks(df['y_smooth'], prominence=np.std(df['y_smooth'])/4, distance=20)
    ar_peaks, _ = find_peaks(df['ar_smooth'], prominence=np.std(df['ar_smooth'])/4, distance=20)

    scored_peaks = []
    for i, y_peak in enumerate(y_peaks):
        y_prominence = y_props['prominences'][i]
        min_dist = min([abs(y_peak - ar_peak) for ar_peak in ar_peaks], default=10)
        score = y_prominence / (min_dist + 1)
        scored_peaks.append({'index': y_peak, 'score': score})
    
    if not scored_peaks: return []

    scored_df = pd.DataFrame(scored_peaks)
    top_10_events = scored_df.nlargest(10, 'score').sort_values('index')
    
    final_peak_data = df.iloc[top_10_events['index']]
    final_events_list = [
        {'frame': row['frame'], 'box': np.fromstring(re.sub(r'\s+', ' ', row['box'].strip('[]')), sep=' ', dtype=int)}
        for _, row in final_peak_data.iterrows()
    ]
    
    print(f"✅ Analysis Complete. Identified {len(final_events_list)} potential hurdle events.")
    return final_events_list

def pass_two_find_finish_line(video_path, trajectory_log_path):
    print("\n--- Starting Pass 2: Finding Finish Line... ---")
    cap = cv2.VideoCapture(video_path)
    traj_df = pd.read_csv(trajectory_log_path)
    athlete_boxes = {
        row['frame']: np.fromstring(re.sub(r'\s+', ' ', row['box'].strip('[]')), sep=' ', dtype=int) 
        for _, row in traj_df.iterrows()
    }

    finish_frame = -1
    frame_number = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_number += 1
        
        if frame_number not in athlete_boxes: continue
        athlete_box = athlete_boxes[frame_number]
        
        if finish_frame == -1:
            results = detection_model(frame, classes=[finish_line_class_id], verbose=False)
            if results[0].boxes.shape[0] > 0:
                finish_line_box = results[0].boxes.xyxy.cpu().numpy()[0]
                athlete_center_x = (athlete_box[0] + athlete_box[2]) / 2
                if athlete_center_x > finish_line_box[0]:
                    finish_frame = frame_number
                    print(f"🏁 Finish Line crossed at frame: {finish_frame}")

    cap.release()
    print(f"✅ Pass 2 Complete.")
    return finish_frame

def pass_three_and_report(video_path, race_start_frame, finish_frame, hurdle_peaks):
    print("\n--- Starting Pass 3: Calculating Final Metrics... ---")
    traj_df = pd.read_csv("pass_1_trajectory_log.csv")
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    # --- Constants for Men's 110m Hurdles ---
    DIST_START_TO_H1 = 13.72
    DIST_INTER_HURDLE = 9.14
    DIST_H10_TO_FINISH = 14.02
    
    box_coords = traj_df['box'].str.extract(r'\[\s*(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s*\]').astype(float)
    traj_df['aspect_ratio'] = (box_coords[2] - box_coords[0]) / (box_coords[3] - box_coords[1])
    
    segment_metrics = []
    flight_time_metrics = []
    identified_hurdles_for_video = []

    # --- Metric Calculations ---

    # 1. Start to Hurdle 1
    time_to_h1 = (hurdle_peaks[0]['frame'] - race_start_frame) / video_fps
    velocity_to_h1 = DIST_START_TO_H1 / time_to_h1
    segment_metrics.append({'Segment': 'Start to H1', 'Time (s)': f"{time_to_h1:.2f}", 'Velocity (m/s)': f"{velocity_to_h1:.2f}"})

    # 2. Hurdle Intervals and Flight Times
    for i, peak in enumerate(hurdle_peaks):
        hurdle_num = i + 1
        peak_frame = peak['frame']
        
        # Flight Time Calculation (using aspect ratio)
        start_window, end_window = peak_frame - POSE_WINDOW_FRAMES, peak_frame + POSE_WINDOW_FRAMES
        window_df = traj_df[(traj_df['frame'] >= start_window) & (traj_df['frame'] <= end_window)]
        takeoff_frame, landing_frame = -1, -1
        for _, row in window_df[window_df['frame'] <= peak_frame].sort_values('frame', ascending=False).iterrows():
            if row['aspect_ratio'] < 1.0: takeoff_frame = row['frame'] + 1; break
        for _, row in window_df[window_df['frame'] >= peak_frame].sort_values('frame', ascending=True).iterrows():
            if row['aspect_ratio'] < 1.0: landing_frame = row['frame'] - 1; break
        
        flight_time = (landing_frame - takeoff_frame) / video_fps if -1 not in [takeoff_frame, landing_frame] and landing_frame > takeoff_frame else 'N/A'
        flight_time_metrics.append({'Hurdle': hurdle_num, 'Flight Time (s)': f"{flight_time:.3f}" if isinstance(flight_time, float) else flight_time})

        # Interval Time and Velocity Calculation
        if i < len(hurdle_peaks) - 1:
            interval_time = (hurdle_peaks[i+1]['frame'] - peak_frame) / video_fps
            interval_velocity = DIST_INTER_HURDLE / interval_time
            segment_metrics.append({'Segment': f'H{i+1} to H{i+2}', 'Time (s)': f"{interval_time:.2f}", 'Velocity (m/s)': f"{interval_velocity:.2f}"})

        # Get hurdle box for validation video
        cap.set(cv2.CAP_PROP_POS_FRAMES, peak_frame - 1)
        ret, frame = cap.read()
        if ret:
            results = detection_model(frame, classes=[hurdle_class_id], verbose=False)
            if results[0].boxes.shape[0] > 0:
                athlete_box_at_peak = peak['box']
                athlete_center = np.array([(athlete_box_at_peak[0] + athlete_box_at_peak[2]) / 2, (athlete_box_at_peak[1] + athlete_box_at_peak[3]) / 2])
                best_hurdle_box = min(results[0].boxes.xyxy.cpu().numpy(), key=lambda h_box: np.linalg.norm(np.array([(h_box[0] + h_box[2]) / 2, (h_box[1] + h_box[3]) / 2]) - athlete_center))
                identified_hurdles_for_video.append({'hurdle_num': hurdle_num, 'peak_frame': peak_frame, 'hurdle_box': best_hurdle_box})

    # 3. Hurdle 10 to Finish
    run_in_time = (finish_frame - hurdle_peaks[9]['frame']) / video_fps
    run_in_velocity = DIST_H10_TO_FINISH / run_in_time
    segment_metrics.append({'Segment': 'H10 to Finish', 'Time (s)': f"{run_in_time:.2f}", 'Velocity (m/s)': f"{run_in_velocity:.2f}"})
    
    cap.release()
    total_race_time = (finish_frame - race_start_frame) / video_fps if race_start_frame > 0 and finish_frame > 0 else -1
    
    # --- Reporting ---
    segment_df = pd.DataFrame(segment_metrics)
    flight_df = pd.DataFrame(flight_time_metrics)

    print("\n--- Race Segment Analysis ---")
    print(segment_df.to_string(index=False))
    print("\n--- Hurdle Clearance Analysis ---")
    print(flight_df.to_string(index=False))
    if total_race_time != -1: print(f"\nTotal Race Time: {total_race_time:.2f} seconds")
    
    # Save combined report
    combined_report = pd.concat([segment_df.set_index('Segment'), flight_df.set_index('Hurdle').rename(columns={'Flight Time (s)': 'Time (s)'})], axis=1)
    combined_report.to_csv("final_metrics_report.csv")
    print("\n✅ Pass 3 Complete. Final report saved.")
    return identified_hurdles_for_video

def generate_validation_video(video_path, trajectory_log_path, final_hurdles):
    print("\n--- Generating Final Validation Video... ---")
    cap = cv2.VideoCapture(video_path)
    width, height, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter('validation_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    traj_df = pd.read_csv(trajectory_log_path)
    athlete_boxes = { row['frame']: np.fromstring(re.sub(r'\s+', ' ', row['box'].strip('[]')), sep=' ', dtype=int) for _, row in traj_df.iterrows() }
    hurdle_events = {h['peak_frame']: h for h in final_hurdles}
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_number += 1
        if frame_number in athlete_boxes:
            box = athlete_boxes[frame_number]
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        if frame_number in hurdle_events:
            hurdle_data = hurdle_events[frame_number]
            box = hurdle_data['hurdle_box'].astype(int)
            label = f"H{hurdle_data['hurdle_num']}"
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 165, 0), 3)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 165, 0), 3)
        out.write(frame)
    cap.release(); out.release()
    print("✅ Validation video saved as 'validation_video.mp4'.")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    video_path = "O2012 F.mp4" 
    trajectory_log_file = "pass_1_trajectory_log.csv"
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}"); exit()
    cv2.namedWindow("Athlete Selection")
    cv2.setMouseCallback("Athlete Selection", select_athlete)
    ret, frame = cap.read()
    if ret:
        while not selection_done:
            display_frame = frame.copy()
            results = detection_model.track(frame, persist=True, verbose=False, tracker='botsort.yaml')
            if results and results[0].boxes.id is not None:
                first_frame_athletes.clear()
                for box, track_id, cls_id in zip(results[0].boxes.xyxy.cpu(), results[0].boxes.id.cpu(), results[0].boxes.cls.cpu()):
                    if int(cls_id) == athlete_class_id:
                        b = box.numpy().astype(int); first_frame_athletes[int(track_id)] = b
                        cv2.rectangle(display_frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                        cv2.putText(display_frame, f"ID: {int(track_id)}", (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(display_frame, "Click on the target athlete, then press 'c'", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            if click_point: cv2.circle(display_frame, click_point, 10, (255, 0, 0), -1)
            cv2.imshow("Athlete Selection", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and click_point:
                for track_id, box in first_frame_athletes.items():
                    if box[0] <= click_point[0] <= box[2] and box[1] <= click_point[1] <= box[3]:
                        target_athlete_id = int(track_id); selection_done = True
                        print(f"✅ Target athlete ID: {target_athlete_id} selected.")
                        break
                if not selection_done: print("No athlete clicked."); click_point = None
            elif key == ord('q'): exit()
    cap.release(); cv2.destroyAllWindows()

    if target_athlete_id is not None:
        start_frame = pass_one_track_athlete(video_path, detection_model, target_athlete_id)
        hurdle_peaks = analyze_trajectory_for_peaks(trajectory_log_file)
        
        if hurdle_peaks:
            finish_frame = pass_two_find_finish_line(video_path, trajectory_log_file)
            identified_hurdles = pass_three_and_report(video_path, start_frame, finish_frame, hurdle_peaks)
            if identified_hurdles:
                 generate_validation_video(video_path, trajectory_log_file, identified_hurdles)