import cv2
import os
import random

def extract_random_frames():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_video_dir_path = os.path.join(BASE_DIR, "Races")
    output_frames_dir_path = os.path.join(BASE_DIR, "Frames")
    input_video_dir = input_video_dir_path
    output_frames_dir = output_frames_dir_path
    
    # Number of frames to extract per video
    num_frames_to_extract = 50
    
    # Supported video file extensions
    video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.flv', '.mpeg')

    # Ensure the main output directory exists
    os.makedirs(output_frames_dir, exist_ok=True)
    print(f"Output directory set to: {output_frames_dir}")

    # Process each file in the input directory
    for file_name in os.listdir(input_video_dir):
        if file_name.lower().endswith(video_extensions):
            video_path = os.path.join(input_video_dir, file_name)
            
            print(f"\nProcessing video: {file_name}...")
            
            # Open the video file for reading
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Error: Could not open video file {video_path}")
                continue

            # Get the total number of frames in the video
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Determine which frame indices to extract
            if total_frames < num_frames_to_extract:
                print(f"⚠️ Warning: Video has only {total_frames} frames. Extracting all of them.")
                frame_indices = range(total_frames)
            else:
                # Generate a list of 50 unique random frame indices
                frame_indices = sorted(random.sample(range(total_frames), num_frames_to_extract))

            frames_saved_count = 0
            # Loop through only the selected random frame indices
            for frame_index in frame_indices:
                # Set the video's position to the desired frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                
                success, frame = cap.read()
                
                if success:
                    # Construct a more descriptive output filename
                    base_name = os.path.splitext(file_name)[0]
                    frame_filename = f"{base_name}_frame_{frame_index:05d}.jpg"
                    save_path = os.path.join(output_frames_dir, frame_filename)
                    
                    # Save the frame as a JPG image
                    cv2.imwrite(save_path, frame)
                    frames_saved_count += 1
            
            # Release the video capture object and clean up
            cap.release()
            print(f"✅ Success! Saved {frames_saved_count} random frames from '{file_name}'")

# Run the main function when the script is executed
if __name__ == "__main__":
    extract_random_frames()