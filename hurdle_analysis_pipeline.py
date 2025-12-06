import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from scipy.signal import find_peaks, savgol_filter
import re
import os

# --- Configuration Constants ---
START_MOTION_THRESHOLD = 13
DYNAMIC_CORRIDOR_WIDTH = 150
POSE_WINDOW_FRAMES = 30    

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_models(detection_model_path=os.path.join(BASE_DIR, "models")):
    """Loads the detection model."""
    print("\n--- Loading detection model... ---")
    try:
        detection_model = YOLO(detection_model_path)
        return detection_model
    except Exception as e:
        print(f"Error loading models: {e}")
        return None

def get_class_ids(detection_model, athlete_class='athlete', hurdle_class='hurdle', finish_class='finish'):
    """Retrieves class IDs from the model."""
    try:
        names_to_ids = {name: i for i, name in detection_model.names.items()}
        class_ids = {
            'athlete': names_to_ids[athlete_class],
            'hurdle': names_to_ids[hurdle_class],
            'finish': names_to_ids[finish_class]
        }
        return class_ids
    except KeyError as e:
        print(f"--- ERROR: Class name {e} not found in model. ---")
        print(f"Actual names found in model: {detection_model.names}")
        return None

def select_athlete_from_video(video_path, detection_model, class_ids):
    """Displays the first frame for user selection and returns the athlete's track ID."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return None, None

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return None, None

    state = {'click_point': None, 'selection_done': False, 'target_athlete_id': None, 'first_frame_athletes': {}}
    
    def _select_athlete_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state['click_point'] = (x, y)

    cv2.namedWindow("Athlete Selection")
    cv2.setMouseCallback("Athlete Selection", _select_athlete_callback)

    while not state['selection_done']:
        display_frame = frame.copy()
        results = detection_model.track(frame, persist=True, verbose=False, tracker='botsort.yaml')
        if results and results[0].boxes.id is not None:
            state['first_frame_athletes'].clear()
            for box, track_id, cls_id in zip(results[0].boxes.xyxy.cpu(), results[0].boxes.id.cpu(), results[0].boxes.cls.cpu()):
                if int(cls_id) == class_ids['athlete']:
                    b = box.numpy().astype(int)
                    state['first_frame_athletes'][int(track_id)] = b
                    cv2.rectangle(display_frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"ID: {int(track_id)}", (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.putText(display_frame, "Click on the target athlete, then press 'c'", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        if state['click_point']: cv2.circle(display_frame, state['click_point'], 10, (255, 0, 0), -1)
        cv2.imshow("Athlete Selection", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and state['click_point']:
            for track_id, box in state['first_frame_athletes'].items():
                if box[0] <= state['click_point'][0] <= box[2] and box[1] <= state['click_point'][1] <= box[3]:
                    state['target_athlete_id'] = int(track_id)
                    state['selection_done'] = True
                    print(f"✅ Target athlete ID: {state['target_athlete_id']} selected.")
                    break
            if not state['selection_done']:
                print("No athlete clicked. Please try again.")
                state['click_point'] = None
        elif key == ord('q'):
            state['target_athlete_id'] = None
            break
            
    cap.release()
    cv2.destroyAllWindows()
    return state['target_athlete_id'], state['first_frame_athletes']


def track_athlete(video_path, detection_model, class_ids, target_athlete_id, first_frame_athletes):
    """Pass 1: Tracks the athlete and returns the trajectory DataFrame and start frame."""
    print("\n--- Starting Pass 1: Tracking athlete's bounding box... ---")
    cap = cv2.VideoCapture(video_path)
    
    trajectory_log = []
    frame_number, race_start_frame = 0, 0
    
    box = first_frame_athletes.get(target_athlete_id)
    starting_block_center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]) if box is not None else None
    last_known_box = box

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_number += 1
        
        results = detection_model.track(frame, persist=True, verbose=False, tracker='botsort.yaml')
        
        all_athletes = []
        if results and results[0].boxes.id is not None:
            all_athletes = [b.numpy().astype(int) for b, c_id in zip(results[0].boxes.xyxy.cpu(), results[0].boxes.cls.cpu()) if int(c_id) == class_ids['athlete']]

        # Find the current box of our target athlete by ID for motion detection
        current_box_of_target = None
        if results and results[0].boxes.id is not None:
            try:
                target_idx = results[0].boxes.id.cpu().tolist().index(target_athlete_id)
                current_box_of_target = results[0].boxes.xyxy.cpu().numpy().astype(int)[target_idx]
            except ValueError:
                pass # ID not found yet

        if race_start_frame == 0 and current_box_of_target is not None and starting_block_center is not None:
            current_center = np.array([(current_box_of_target[0] + current_box_of_target[2]) / 2, (current_box_of_target[1] + current_box_of_target[3]) / 2])
            if np.linalg.norm(current_center - starting_block_center) > START_MOTION_THRESHOLD:
                race_start_frame = frame_number
                print(f"🚀 Motion detected! Race starts at frame: {race_start_frame}")
        
        if race_start_frame > 0:
            if last_known_box is not None:
                center_x = (last_known_box[0] + last_known_box[2]) / 2
                corridor = (center_x - DYNAMIC_CORRIDOR_WIDTH / 2, center_x + DYNAMIC_CORRIDOR_WIDTH / 2)
                candidates = [b for b in all_athletes if corridor[0] < ((b[0] + b[2]) / 2) < corridor[1]]
                
                if candidates:
                    last_center = np.array([(last_known_box[0] + last_known_box[2]) / 2, (last_known_box[1] + last_known_box[3]) / 2])
                    last_known_box = min(candidates, key=lambda b: np.linalg.norm(np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]) - last_center))

            if last_known_box is not None:
                trajectory_log.append({'frame': frame_number, 'box': np.array2string(last_known_box)})

    cap.release()
    trajectory_df = pd.DataFrame(trajectory_log)
    print(f"✅ Pass 1 Complete. Logged {len(trajectory_df)} data points.")
    return trajectory_df, race_start_frame

def find_hurdle_peaks(trajectory_df):
    """Analyzes trajectory DataFrame to find 10 hurdle peaks."""
    print("\n--- Analyzing Trajectory: Using Enhanced Two-Factor Peak Detection... ---")
    if trajectory_df.empty: 
        print("--- ERROR: Trajectory DataFrame is empty. ---")
        return []
        
    box_coords = trajectory_df['box'].str.extract(r'\[\s*(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s*\]').astype(float)
    
    # Check for empty or all-NaN box_coords
    if box_coords.isnull().all().all():
        print("--- ERROR: Could not parse any box coordinates from trajectory. ---")
        return []

    # Calculate Aspect Ratio (Width / Height)
    trajectory_df['aspect_ratio'] = (box_coords[2] - box_coords[0]) / (box_coords[3] - box_coords[1])
    trajectory_df['y_center'] = (box_coords[1] + box_coords[3]) / 2
    
    # Invert Y for peak finding (higher on screen = lower Y value)
    inverted_y = -trajectory_df['y_center']
    
    # Ensure window length is valid (odd and smaller than data)
    window_length = min(21, len(trajectory_df))
    if window_length % 2 == 0: window_length -= 1
    if window_length < 5: 
        print("--- ERROR: Not enough data points for smoothing. ---")
        return []

    # Apply smoothing
    trajectory_df['y_smooth'] = savgol_filter(inverted_y, window_length, 3) 
    trajectory_df['ar_smooth'] = savgol_filter(trajectory_df['aspect_ratio'], window_length, 3)

    # Calculate separate prominence thresholds for each signal
    # Use a small epsilon to prevent zero standard deviation
    y_prominence_threshold = (np.std(trajectory_df['y_smooth']) / 8) + 1e-6
    ar_prominence_threshold = (np.std(trajectory_df['ar_smooth']) / 4) + 1e-6
    
    print(f"Calculated Y-Prominence Threshold: {y_prominence_threshold:.4f}")
    print(f"Calculated AR-Prominence Threshold: {ar_prominence_threshold:.4f}")

    # Find peaks using their respective thresholds
    y_peaks, y_props = find_peaks(trajectory_df['y_smooth'], prominence=y_prominence_threshold, distance=20)
    ar_peaks, _ = find_peaks(trajectory_df['ar_smooth'], prominence=ar_prominence_threshold, distance=20)

    print(f"Found {len(y_peaks)} raw vertical peaks and {len(ar_peaks)} raw aspect ratio peaks.")
    
    scored_peaks = []
    
    if len(ar_peaks) == 0:
        print("--- WARNING: No AR peaks found. Falling back to Y-peaks only. Results may be inaccurate. ---")
        for i, y_peak in enumerate(y_peaks):
            scored_peaks.append({'index': y_peak, 'score': y_props['prominences'][i]})
    else:
        for i, y_peak in enumerate(y_peaks):
            min_dist = min([abs(y_peak - ar_peak) for ar_peak in ar_peaks], default=10)
            scored_peaks.append({'index': y_peak, 'score': y_props['prominences'][i] / (min_dist + 1)})

    if not scored_peaks: 
        print("--- ERROR: No peaks scored. ---")
        return []
        
    top_10_events = pd.DataFrame(scored_peaks).nlargest(10, 'score').sort_values('index')
    
    final_events_list = [
        {'frame': row['frame'], 'box': np.fromstring(re.sub(r'\s+', ' ', row['box'].strip('[]')), sep=' ', dtype=int)}
        for _, row in trajectory_df.iloc[top_10_events['index']].iterrows()
    ]
    
    print(f"✅ Analysis Complete. Identified {len(final_events_list)} potential hurdle events.")
    return final_events_list

def find_finish_line_cross(video_path, detection_model, class_ids, trajectory_df, finish_buffer_percent=0.25):
    """
    Pass 2: Finds the finish line crossing frame.
    Includes a buffer to prevent premature detection.
    """
    print(f"\n--- Starting Pass 2: Finding Finish Line (with {finish_buffer_percent*100}% buffer)... ---")
    cap = cv2.VideoCapture(video_path)
    athlete_boxes = { row['frame']: np.fromstring(re.sub(r'\s+', ' ', row['box'].strip('[]')), sep=' ', dtype=int) for _, row in trajectory_df.iterrows() }
    finish_frame = -1
    frame_number = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_number += 1
        
        if frame_number not in athlete_boxes: continue
            
        if finish_frame == -1:
            results = detection_model(frame, classes=[class_ids['finish']], verbose=False)
            
            if results and results[0].boxes.shape[0] > 0:
                finish_line_box = results[0].boxes.xyxy.cpu().numpy()[0]
                
                finish_line_width = finish_line_box[2] - finish_line_box[0]
                trigger_point = finish_line_box[0] + (finish_line_width * finish_buffer_percent)
                
                athlete_box = athlete_boxes[frame_number]
                athlete_center_x = (athlete_box[0] + athlete_box[2]) / 2
                
                if athlete_center_x > trigger_point:
                    finish_frame = frame_number
                    print(f"🏁 Finish Line crossed at frame: {finish_frame} (Buffer Trigger: {trigger_point:.2f}px)")
                    break
    
    cap.release()
    print(f"✅ Pass 2 Complete.")
    return finish_frame

def calculate_metrics_and_report(video_path, detection_model, class_ids, race_start_frame, finish_frame, hurdle_peaks, trajectory_df):
    """Pass 3: Calculates all metrics and returns them as DataFrames."""
    print("\n--- Starting Pass 3: Calculating Final Metrics... ---")
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    DISTANCES = {'START_TO_H1': 13.72, 'INTER_HURDLE': 9.14, 'H10_TO_FINISH': 14.02}
    segment_metrics, flight_time_metrics, identified_hurdles = [], [], []
    
    if not hurdle_peaks or len(hurdle_peaks) < 10:
        print("--- ERROR: Not enough hurdle peaks found to calculate metrics. ---")
        cap.release()
        return pd.DataFrame(), pd.DataFrame(), np.nan, []
        
    time_to_h1 = (hurdle_peaks[0]['frame'] - race_start_frame) / video_fps
    segment_metrics.append({'Segment': 'Start to H1', 'Time (s)': time_to_h1, 'Velocity (m/s)': DISTANCES['START_TO_H1'] / time_to_h1})
    
    for i, peak in enumerate(hurdle_peaks):
        start_w, end_w = peak['frame'] - POSE_WINDOW_FRAMES, peak['frame'] + POSE_WINDOW_FRAMES
        window_df = trajectory_df[(trajectory_df['frame'] >= start_w) & (trajectory_df['frame'] <= end_w)]
        takeoff, landing = -1, -1
        
        if 'aspect_ratio' not in window_df.columns and not window_df.empty:
            box_coords = window_df['box'].str.extract(r'\[\s*(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s*\]').astype(float)
            window_df['aspect_ratio'] = (box_coords[2] - box_coords[0]) / (box_coords[3] - box_coords[1])

        for _, row in window_df[window_df['frame'] <= peak['frame']].sort_values('frame', ascending=False).iterrows():
            if 'aspect_ratio' in row and row['aspect_ratio'] < 1.0: takeoff = row['frame'] + 1; break
        for _, row in window_df[window_df['frame'] >= peak['frame']].sort_values('frame').iterrows():
            if 'aspect_ratio' in row and row['aspect_ratio'] < 1.0: landing = row['frame'] - 1; break
            
        flight_time = (landing - takeoff) / video_fps if -1 not in [takeoff, landing] and landing > takeoff else np.nan
        flight_time_metrics.append({'Hurdle': i + 1, 'Flight Time (s)': flight_time})
        
        if i < len(hurdle_peaks) - 1:
            interval = (hurdle_peaks[i+1]['frame'] - peak['frame']) / video_fps
            segment_metrics.append({'Segment': f'H{i+1} to H{i+2}', 'Time (s)': interval, 'Velocity (m/s)': DISTANCES['INTER_HURDLE'] / interval})
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, peak['frame'] - 1)
        ret, frame = cap.read()
        if ret:
            results = detection_model(frame, classes=[class_ids['hurdle']], verbose=False)
            if results and results[0].boxes.shape[0] > 0:
                center = np.array([(peak['box'][0] + peak['box'][2]) / 2, (peak['box'][1] + peak['box'][3]) / 2])
                best_box = min(results[0].boxes.xyxy.cpu().numpy(), key=lambda b: np.linalg.norm(np.array([(b[0]+b[2])/2, (b[1]+b[3])/2]) - center))
                identified_hurdles.append({'hurdle_num': i + 1, 'peak_frame': peak['frame'], 'hurdle_box': best_box})
    
    if finish_frame > 0 and hurdle_peaks:
        run_in_time = (finish_frame - hurdle_peaks[-1]['frame']) / video_fps
        segment_metrics.append({'Segment': 'H10 to Finish', 'Time (s)': run_in_time, 'Velocity (m/s)': DISTANCES['H10_TO_FINISH'] / run_in_time})
    else:
        segment_metrics.append({'Segment': 'H10 to Finish', 'Time (s)': np.nan, 'Velocity (m/s)': np.nan})

    cap.release()
    
    total_time = (finish_frame - race_start_frame) / video_fps if race_start_frame > 0 and finish_frame > 0 else np.nan
    
    segment_df = pd.DataFrame(segment_metrics).round(2)
    flight_df = pd.DataFrame(flight_time_metrics).round(3)
    
    print("\n--- Race Segment Analysis ---")
    print(segment_df.to_string(index=False))
    print("\n--- Hurdle Clearance Analysis ---")
    print(flight_df.to_string(index=False))
    if not np.isnan(total_time): print(f"\nTotal Race Time: {total_time:.2f} seconds")
    
    print("\n✅ Pass 3 Complete. Final report data generated.")
    return segment_df, flight_df, total_time, identified_hurdles

def generate_validation_video(video_path, trajectory_df, final_hurdles, output_path="validation_video.mp4"):
    """Generates a validation video with athlete and hurdle boxes."""
    print("\n--- Generating Final Validation Video... ---")
    cap = cv2.VideoCapture(video_path)
    width, height, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    athlete_boxes = { row['frame']: np.fromstring(re.sub(r'\s+', ' ', row['box'].strip('[]')), sep=' ', dtype=int) for _, row in trajectory_df.iterrows() }
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
            data = hurdle_events[frame_number]
            box = data['hurdle_box'].astype(int)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 165, 0), 3)
            cv2.putText(frame, f"H{data['hurdle_num']}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 165, 0), 3)
            
        out.write(frame)
        
    cap.release(); out.release()
    print(f"✅ Validation video saved as '{output_path}'.")