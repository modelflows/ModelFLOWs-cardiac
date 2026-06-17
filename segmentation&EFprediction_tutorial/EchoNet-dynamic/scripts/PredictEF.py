import os
import glob
import cv2
import torch
import torchvision
import numpy as np
import csv

# ================= UPDATE PATHS HERE =================
# Replace the leading "..." with the folder that holds EchoNet-dynamic on your
# machine (e.g. r"D:\projects\EchoNet-dynamic\demo\video"). Keep everything from
# "EchoNet-dynamic" onward exactly as written.
INPUT_DIR  = r"...\EchoNet-dynamic\demo\video"                       # input videos
WEIGHTS    = r"...\EchoNet-dynamic\stats\r2plus1d_18_32_2_pretrained.pt"               # model weights (.pt/.pth)
OUTPUT_CSV = r"...\EchoNet-dynamic\demo_output\EFprediction.csv"     # output CSV
# =====================================================

# Per-channel mean/std for input normalization (0-1 scale).
# These values were computed from the EchoNet-Dynamic training set.
# Replace them if you compute stats from your own data (see CalculateStats.py).
MEAN = [0.128, 0.129, 0.130]
STD  = [0.196, 0.196, 0.197]

class FolderVideoDataset(torch.utils.data.Dataset):
    """
    Custom Dataset: Directly reads videos from a folder, bypassing FileList.csv
    """
    def __init__(self, video_dir, frames=32, period=2):
        self.video_paths = glob.glob(os.path.join(video_dir, "*.avi")) + \
                           glob.glob(os.path.join(video_dir, "*.mp4"))
        self.frames = frames
        self.period = period
        
        # Mean/std for normalization (0-1 scale), see MEAN/STD at the top of the file
        self.mean = np.array(MEAN)
        self.std = np.array(STD)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        file_name = os.path.basename(video_path)

        # 1. Read all frames using OpenCV
        cap = cv2.VideoCapture(video_path)
        frames_list = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            resized = cv2.resize(rgb, (112, 112))
            frames_list.append(resized)
        cap.release()

        video_tensor = np.array(frames_list) # Shape: (TotalFrames, 112, 112, 3)
        T = len(video_tensor)
        target_length = self.frames * self.period
        clips = []

        # 2. Faithfully reproduce the official Test-Time Augmentation (frame-by-frame sliding)
        if T >= target_length:
            # Official approach: slice frame by frame, extracting every possible clip
            for start in range(T - target_length + 1):
                clip = video_tensor[start : start + target_length : self.period]
                clips.append(clip)
        else:
            # If the video is too short, pad it and treat it as a single clip
            indices = np.arange(0, target_length, self.period) % max(T, 1)
            clips.append(video_tensor[indices])

        # 3. Official data preprocessing pipeline (vectorized for speed)
        # Scale to 0-1, then normalize with the EchoNet training-set mean/std
        clips = np.array(clips).astype(np.float32) / 255.0
        clips = (clips - self.mean) / self.std

        # Reorder dimensions: (Num_clips, H, W, C) -> (Num_clips, C, F, H, W)
        clips = np.transpose(clips, (0, 4, 1, 2, 3)) 

        return torch.tensor(clips, dtype=torch.float32), file_name


def main(data_dir, weights_path, output_csv, device_name='cuda'):
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for inference...")

    # 1. Initialize the model architecture
    model = torchvision.models.video.r2plus1d_18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)

    # 2. Load weights
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    
    model.to(device)
    model.eval()

    # 3. Load custom dataset
    dataset = FolderVideoDataset(data_dir)
    if len(dataset) == 0:
        print("Error: No .avi or .mp4 videos found in the specified folder!")
        return

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"Successfully located {len(dataset)} videos. Starting prediction...")

    # Initialize results list with headers (keeping your CSV format)
    results_list = [["FileName", "Predicted_EF(%)", "Status"]]

    # 4. Run prediction
    # Set an OOM-safe chunk size; if your VRAM exceeds 12GB, you can raise it to 20 or even 40 to speed things up
    block_size = 10

    with torch.no_grad():
        for inputs, file_names in dataloader:
            # Drop the batch dimension; shape becomes (Num_clips, Channel, Frames, Height, Width)
            inputs = inputs.squeeze(0)
            file_name = file_names[0]
            n_clips = inputs.size(0)

            pred_list = []

            # [Key change]: feed clips to the GPU in batches like the official code, to avoid VRAM overflow!
            for j in range(0, n_clips, block_size):
                chunk = inputs[j : j + block_size].to(device, dtype=torch.float32)
                outputs = model(chunk)
                # Collect this batch's predictions
                pred_list.extend(outputs.view(-1).cpu().numpy().tolist())

            # Average the predictions across all clips of this long video
            pred_ef = np.mean(pred_list)

            if pred_ef < 50:
                status = "Low"
            elif pred_ef <= 70:
                status = "Normal"
            else:
                status = "High"
            print(f"[{file_name}] -> contains {n_clips} clips, overall predicted EF: {pred_ef:.2f}% ({status})")

            # Keep your logic: write only one row with the final average
            results_list.append([file_name, round(pred_ef, 2), status])

    # 5. Export results to CSV
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    print(f"\nSaving prediction results to: {output_csv}")
    with open(output_csv, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerows(results_list)
    print("✅ Save complete!")

if __name__ == "__main__":
    main(data_dir=INPUT_DIR, weights_path=WEIGHTS, output_csv=OUTPUT_CSV)