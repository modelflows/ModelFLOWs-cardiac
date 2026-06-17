import cv2
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

# ================= UPDATE PATHS HERE =================
# Replace the leading "..." with the folder that holds EchoNet-dynamic on your
# machine. INPUT_DIR points at the bundled demo videos; for real normalization
# stats, point it at your full EchoNet-Dynamic "Videos" folder and (optionally)
# set CSV_PATH to its FileList.csv — only the TRAIN split is then used.
INPUT_DIR = r"...\EchoNet-dynamic\demo\video"   # folder of videos to scan
CSV_PATH  = None                                # or r"...\FileList.csv" (TRAIN split only)
# =====================================================

def get_video_paths(video_dir, csv_path):
    """
    Decides whether to return a filtered list of training videos based on a CSV file
    or to return all videos in the directory.
    """
    
    # Branch 1: CSV path provided and file exists
    if csv_path and os.path.exists(csv_path):
        print(f"📄 CSV file detected: {csv_path}")
        try:
            df = pd.read_csv(csv_path)
            if 'Split' not in df.columns or 'FileName' not in df.columns:
                print("⚠️ CSV missing 'FileName' or 'Split' columns. Switching to [Full Scan Mode]...")
                return get_all_videos(video_dir), "Full Dataset"

            # Filter for rows where Split is 'TRAIN'
            train_df = df[df['Split'].str.upper() == 'TRAIN']
            train_filenames = train_df['FileName'].tolist()

            if not train_filenames:
                print("⚠️ No rows marked as 'TRAIN' found in CSV. Switching to [Full Scan Mode]...")
                return get_all_videos(video_dir), "Full Dataset"

            print(f"🎯 Found {len(train_filenames)} training videos in CSV. Verifying local files...")
            valid_paths = []
            for fname in train_filenames:
                # Ensure filename has an extension
                if not fname.lower().endswith(('.avi', '.mp4')):
                    fname += '.avi'
                
                full_path = os.path.join(video_dir, fname)
                if os.path.exists(full_path):
                    valid_paths.append(full_path)
            
            return valid_paths, "Training Set Only"
            
        except Exception as e:
            print(f"⚠️ Error reading CSV ({e}). Switching to [Full Scan Mode]...")
            return get_all_videos(video_dir), "Full Dataset"
            
    # Branch 2: No CSV provided or path is invalid
    else:
        if csv_path:
            print(f"⚠️ Warning: Cannot find CSV file at {csv_path}")
        print("🌍 CSV filtering not used. Entering [Full Scan Mode].")
        return get_all_videos(video_dir), "Full Dataset"

def get_all_videos(video_dir):
    """Retrieves all .avi and .mp4 files in the directory."""
    return glob.glob(os.path.join(video_dir, "*.avi")) + glob.glob(os.path.join(video_dir, "*.mp4"))


def calculate_stats_auto(video_dir, csv_path):
    # 1. Get video list automatically
    valid_video_paths, mode_name = get_video_paths(video_dir, csv_path)

    if not valid_video_paths:
        print(f"❌ No valid video files found in {video_dir}!")
        return

    print(f"\n📁 Final count: {len(valid_video_paths)} videos ready for processing. Starting calculation...")

    # 2. Core statistical calculation
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sum_squared = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for video_path in tqdm(valid_video_paths, desc="Processing Videos", unit="video"):
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret: 
                break
            
            # Preprocessing: BGR to RGB -> Resize -> Normalize to [0, 1]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (112, 112))
            frame = frame.astype(np.float32) / 255.0

            # Cumulative sum for mean and std calculation
            channel_sum += np.sum(frame, axis=(0, 1))
            channel_sum_squared += np.sum(frame ** 2, axis=(0, 1))
            pixel_count += frame.shape[0] * frame.shape[1]
            
        cap.release()

    if pixel_count == 0:
        print("❌ All videos are empty. Calculation failed!")
        return

    # 3. Calculate and display results
    mean = channel_sum / pixel_count
    # Standard deviation formula: sqrt(E[X^2] - (E[X])^2)
    std = np.sqrt((channel_sum_squared / pixel_count) - (mean ** 2))

    print("\n" + "="*60)
    print(f"🎓 Normalization parameters for [{mode_name}] calculated!")
    print("="*60)
    print("Copy the following lines into your prediction script (PredictEF.py / PredictSegmentation.py):")
    print("-" * 60)
    print(f"MEAN = [{mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.3f}]")
    print(f"STD  = [{std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f}]")
    print("="*60 + "\n")

if __name__ == "__main__":
    calculate_stats_auto(INPUT_DIR, CSV_PATH)