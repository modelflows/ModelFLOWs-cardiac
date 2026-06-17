# Zhuoqun Zhao 5/5/2026
# All-in-One Version: Auto-detects Videos and .npy array files.
import torch
import torchvision
import cv2
import numpy as np
import os
import glob
import scipy.signal
import matplotlib.pyplot as plt
import skimage.draw
from tqdm import tqdm
import echonet

# ================= UPDATE PATHS HERE =================
# Replace the leading "..." with the folder that holds EchoNet-dynamic on your
# machine (e.g. r"D:\projects\EchoNet-dynamic\demo\video"). Keep everything from
# "EchoNet-dynamic" onward exactly as written.
INPUT_DIR  = r"...\EchoNet-dynamic\demo\video"                    # input videos and/or .npy frames
WEIGHTS    = r"...\EchoNet-dynamic\stats\deeplabv3_resnet50_random.pt"  # model weights (.pt/.pth)
OUTPUT_DIR = r"...\EchoNet-dynamic\demo_output"                   # main output folder
# =====================================================

# Per-channel mean/std for input normalization (0-1 scale).
# These values were computed from the EchoNet-Dynamic training set.
# Replace them if you compute stats from your own data (see CalculateStats.py).
MEAN = [0.128, 0.129, 0.130]
STD  = [0.196, 0.196, 0.197]


def predict_all_in_one(input_dir, weights, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Running device: {device}")

    # Create the main output directory; subfolders are created on demand so no empty folders are left behind
    os.makedirs(output_dir, exist_ok=True)
    video_out_dir = os.path.join(output_dir, "segmentation_videos")
    plot_dir = os.path.join(output_dir, "area_curves")
    mask_dir = os.path.join(output_dir, "segmentation_images")

    # 1. Load the model
    if torch.cuda.is_available():
        print("🧠 Loading DeepLabV3 model onto GPU...")
    else:
        print("⚠️ No GPU detected, loading model on CPU (may be slow)...")

    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, aux_loss=False)
    model.classifier[-1] = torch.nn.Conv2d(model.classifier[-1].in_channels, 1, kernel_size=model.classifier[-1].kernel_size)

    checkpoint = torch.load(weights, map_location=device)
    state_dict = checkpoint['state_dict']
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # 2. Scan and categorize files
    all_files = glob.glob(os.path.join(input_dir, "*"))
    video_files = [f for f in all_files if f.lower().endswith(('.avi', '.mp4'))]
    npy_files = [f for f in all_files if f.lower().endswith('.npy')]

    print(f"\n📁 Scan complete: found {len(video_files)} videos and {len(npy_files)} .npy files.")

    # ==========================================
    # Module A: Array Processing (.npy files)
    # ==========================================
    if npy_files:
        print("\n🧮 Starting to process .npy matrix data...")
        os.makedirs(mask_dir, exist_ok=True)
        npy_csv_path = os.path.join(output_dir, "image_lv_area.csv")
        with open(npy_csv_path, "w", encoding='utf-8') as npy_csv:
            npy_csv.write("Filename,LV_Area_Pixels(112x112_scale),Status\n")

            for npy_path in tqdm(npy_files, desc="NPY progress", unit="file"):
                filename = os.path.basename(npy_path)

                try:
                    # Load the NumPy matrix
                    img_array = np.load(npy_path)
                except Exception as e:
                    print(f"\n⚠️ Unable to read {filename}: {e}")
                    continue

                orig_h, orig_w = img_array.shape[0], img_array.shape[1]

                # --- NPY format robustness: uniformly convert to a (H, W, 3) uint8 image with value range 0-255 ---
                # 1. Normalize to 0-255 (eases subsequent cv2 processing and visualization output)
                _min, _max = img_array.min(), img_array.max()
                if _max > _min:
                    img_norm = ((img_array - _min) / (_max - _min) * 255.0).astype(np.uint8)
                else:
                    img_norm = np.zeros_like(img_array, dtype=np.uint8)

                # 2. Align the channel dimension
                if img_norm.ndim == 2:
                    # Single-channel grayscale (H, W) -> convert to three channels (H, W, 3)
                    img_rgb = np.stack((img_norm,)*3, axis=-1)
                elif img_norm.ndim == 3:
                    if img_norm.shape[0] in [1, 3]:
                        # Channel-first format (C, H, W) -> convert to channel-last (H, W, C)
                        img_norm = np.transpose(img_norm, (1, 2, 0))

                    if img_norm.shape[2] == 1:
                        # (H, W, 1) -> convert to (H, W, 3)
                        img_rgb = np.concatenate([img_norm]*3, axis=-1)
                    else:
                        # Assume it is already (H, W, 3)
                        img_rgb = img_norm
                else:
                    print(f"\n⚠️ {filename} has an unsupported dimensionality, skipping.")
                    continue
                # -------------------------------------------------------------

                # Preprocess to 112x112
                img_resized = cv2.resize(img_rgb, (112, 112))
                input_tensor = torch.tensor(img_resized).float() / 255.0
                input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)

                # Apply input normalization (prevents the domain shift that degrades segmentation)
                mean = torch.tensor(MEAN).view(1, 3, 1, 1)
                std = torch.tensor(STD).view(1, 3, 1, 1)
                input_tensor = (input_tensor - mean) / std
                input_tensor = input_tensor.to(device)

                # Inference and area computation
                with torch.no_grad():
                    output = model(input_tensor)["out"]
                    pred = output[0, 0].cpu().numpy()

                mask = (pred > 0).astype(np.uint8)
                area = np.sum(mask)

                # ------ Build a video-like side-by-side layout ------
                masked_img = img_resized.copy()
                # Highlight the mask region by setting channel 1 (green channel) to 255
                masked_img[mask > 0, 1] = 255

                # Horizontal concatenation: original + masked (shape: 112, 224, 3)
                side_by_side = np.concatenate((img_resized, masked_img), axis=1)

                # Bottom canvas (shape: 112, 224, 3)
                blank_canvas = np.zeros_like(side_by_side)

                # Print the area text centered on the bottom canvas
                text = f"LV Area: {area} px"
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(text, font, 0.6, 1)[0]
                text_x = (blank_canvas.shape[1] - text_size[0]) // 2
                text_y = (blank_canvas.shape[0] + text_size[1]) // 2
                cv2.putText(blank_canvas, text, (text_x, text_y), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                # Vertical concatenation: image + bottom data panel (shape: 224, 224, 3)
                img_combined = np.concatenate((side_by_side, blank_canvas), axis=0)

                # Output is RGB by default. To save as a normally-colored png with cv2, convert to BGR first
                final_bgr = cv2.cvtColor(img_combined, cv2.COLOR_RGB2BGR)

                # Upscale 2x (448x448) for easier visual inspection, using nearest-neighbor to keep it crisp
                final_bgr_large = cv2.resize(final_bgr, (448, 448), interpolation=cv2.INTER_NEAREST)

                # Save the final image
                mask_filename = f"{os.path.splitext(filename)[0]}_segmentation.png"
                cv2.imwrite(os.path.join(mask_dir, mask_filename), final_bgr_large)

                npy_csv.write(f"{filename},{area},Success\n")

    # ==========================================
    # Module B: Video Processing (unchanged)
    # ==========================================
    if video_files:
        print("\n🎞️ Starting to process video sequences (using EchoNet global normalization)...")
        os.makedirs(video_out_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)
        vid_csv_path = os.path.join(output_dir, "video_lv_area.csv")
        with open(vid_csv_path, "w", encoding='utf-8') as vid_csv:
            vid_csv.write("Filename,Frame,Size,ComputerSystole\n")

            for video_path in tqdm(video_files, desc="Video progress", unit="vid"):
                filename = os.path.basename(video_path)
                out_filename = os.path.splitext(filename)[0] + "_segmentation.avi"
                output_path = os.path.join(video_out_dir, out_filename)

                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (112, 112))
                    frames.append(frame)
                cap.release()

                if not frames: continue

                orig_frames = np.array(frames)
                video_tensor = torch.tensor(orig_frames).float() / 255.0
                video_tensor = video_tensor.permute(0, 3, 1, 2)

                mean = torch.tensor(MEAN).view(1, 3, 1, 1)
                std = torch.tensor(STD).view(1, 3, 1, 1)
                video_tensor = (video_tensor - mean) / std
                video_tensor = video_tensor.to(device)

                batch_size = 20
                preds = []
                with torch.no_grad():
                    for i in range(0, len(video_tensor), batch_size):
                        batch = video_tensor[i:i+batch_size]
                        out = model(batch)["out"]
                        preds.append(out.cpu().numpy())
                preds = np.concatenate(preds, axis=0)

                logits = preds[:, 0, :, :]
                sizes = (logits > 0).sum(axis=(1, 2))

                systole = set()
                if len(sizes) > 10:
                    trim_min = sorted(sizes)[round(len(sizes) ** 0.05)]
                    trim_max = sorted(sizes)[round(len(sizes) ** 0.95)]
                    trim_range = trim_max - trim_min
                    systole = set(scipy.signal.find_peaks(-sizes, distance=20, prominence=(0.50 * trim_range))[0])

                for frame_idx, s in enumerate(sizes):
                    is_systole = 1 if frame_idx in systole else 0
                    vid_csv.write(f"{filename},{frame_idx},{s},{is_systole}\n")

                fig = plt.figure(figsize=(len(sizes) / fps * 1.5, 3))
                plt.scatter(np.arange(len(sizes)) / fps, sizes, s=1)
                ylim = plt.ylim()
                for s in systole:
                    plt.plot(np.array([s, s]) / fps, ylim, linewidth=1, color='gray', linestyle='--')
                plt.ylim(ylim)
                plt.title(os.path.splitext(filename)[0])
                plt.xlabel("Seconds")
                plt.ylabel("Size (pixels)")
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f"{os.path.splitext(filename)[0]}_area_curve.pdf"))
                plt.close(fig)

                masked_frames = orig_frames.copy()
                for f_idx in range(len(orig_frames)):
                    mask = logits[f_idx] > 0
                    masked_frames[f_idx, mask, 0] = 255  # Highlight the segmented region in the RGB red channel

                side_by_side = np.concatenate((orig_frames, masked_frames), axis=2)
                blank_canvas = np.zeros_like(side_by_side)
                video_combined = np.concatenate((side_by_side, blank_canvas), axis=1)

                norm_sizes = sizes.astype(float).copy()
                if norm_sizes.max() > norm_sizes.min():
                    norm_sizes -= norm_sizes.min()
                    norm_sizes = norm_sizes / norm_sizes.max()
                norm_sizes = 1 - norm_sizes

                for f_idx, s in enumerate(norm_sizes):
                    y_pos = int(round(115 + 100 * s))
                    x_pos = int(round(f_idx / len(sizes) * 200 + 10))
                    video_combined[:, y_pos, x_pos] = [255, 255, 255]

                    if f_idx in systole:
                        video_combined[:, 115:224, x_pos] = [150, 150, 150]

                    r, c = skimage.draw.disk((y_pos, x_pos), 4.1)
                    r = np.clip(r, 0, 223)
                    c = np.clip(c, 0, 223)
                    video_combined[f_idx, r, c] = [255, 255, 255]

                final_video = video_combined.astype(np.uint8).transpose(3, 0, 1, 2)
                echonet.utils.savevideo(output_path, final_video, int(fps))

    print(f"\n✅ All processing complete!")
    print(f"📂 Main output directory: {os.path.abspath(output_dir)}")
    if video_files:
        print(f"   🎥 Segmentation videos & area curves saved in: segmentation_videos/ & area_curves/")
        print(f"   📊 Per-frame LV area log saved in: video_lv_area.csv")
    if npy_files:
        print(f"   🧮 Segmentation images saved in: segmentation_images/")
        print(f"   📈 Per-image LV area log saved in: image_lv_area.csv")

if __name__ == "__main__":
    # Work around the Windows console (cp1252) being unable to print emoji
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    predict_all_in_one(INPUT_DIR, WEIGHTS, OUTPUT_DIR)
