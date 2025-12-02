# Import the functions from your new pipeline library
import hurdle_analysis_pipeline as hap
import pandas as pd
import os

def run_full_analysis(video_path, generate_video=False):
    """
    Orchestrates the full analysis pipeline for a given video.
    
    Args:
        video_path (str): The path to the video file.
        generate_video (bool): If True, a validation video will be saved.
    
    Returns:
        tuple: A tuple containing the segment_df, flight_df, and total_time.
    """
    # 1. Setup
    detection_model = hap.load_models()
    if detection_model is None: return None, None, None

    class_ids = hap.get_class_ids(detection_model)
    if class_ids is None: return None, None, None

    # 2. User Interaction
    target_id, first_frame_athletes = hap.select_athlete_from_video(video_path, detection_model, class_ids)
    if target_id is None:
        print("No athlete selected. Exiting.")
        return None, None, None

    # 3. Core Analysis
    trajectory_df, start_frame = hap.track_athlete(video_path, detection_model, class_ids, target_id, first_frame_athletes)
    
    hurdle_peaks = hap.find_hurdle_peaks(trajectory_df)
    if not hurdle_peaks:
        print("Could not identify hurdle peaks. Analysis cannot continue.")
        return None, None, None

    finish_frame = hap.find_finish_line_cross(video_path, detection_model, class_ids, trajectory_df)
    
    # 4. Reporting and Validation
    segment_df, flight_df, total_time, identified_hurdles = hap.calculate_metrics_and_report(
        video_path, detection_model, class_ids, start_frame, finish_frame, hurdle_peaks, trajectory_df
    )
    
    # 5. Optional Video Generation
    if generate_video and identified_hurdles:
        hap.generate_validation_video(video_path, trajectory_df, identified_hurdles)
    
    print("\n--- Pipeline Complete ---")
    return segment_df, flight_df, total_time


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    VIDEO_FILE_path = os.path.join(BASE_DIR, "races", "O2012 F.mp4")
    VIDEO_FILE =  VIDEO_FILE_path
    
    # Run the analysis
    segments, flights, time = run_full_analysis(VIDEO_FILE, generate_video=True)
    
    # You can now work with the final DataFrames here
    if segments is not None:
        print("\n--- Final DataFrames available in main script ---")
        print("\nSegment Analysis DataFrame:")
        print(segments)