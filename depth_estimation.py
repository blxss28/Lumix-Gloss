import torch
import cv2
import numpy as np

# Load MiDaS model (recommended DPT-Hybrid or DPT-Large)
def load_model():
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")  # Use "DPT_Large" for higher quality
    midas.eval()
    return midas

# Load transforms for DPT-Hybrid/DPT-Large (handles resizing, normalization, etc.)
def load_transforms():
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    return midas_transforms.dpt_transform  # Use appropriate transform for DPT models

# Estimate depth from a single frame
def estimate_depth(model, transform, frame):
    # Apply MiDaS transforms
    input_tensor = transform(frame).unsqueeze(0)

    # Disable gradient computation for inference
    with torch.no_grad():
        depth = model(input_tensor)

    # Convert depth tensor to numpy array
    depth_map = depth.squeeze().cpu().numpy()

    return depth_map

# Normalize and save depth map for visualization
def save_normalized_depth_map(depth_map, output_path):
    normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(output_path, normalized.astype(np.uint8))
    print(f"Depth map saved to '{output_path}'")

# Main function for real-time depth estimation using webcam
def main():
    # Load model and transforms
    model = load_model()
    transform = load_transforms()

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    print("Press 'q' to quit the application.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from webcam.")
            break

        # Convert frame to RGB (as required by MiDaS)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Estimate depth
        depth_map = estimate_depth(model, transform, frame_rgb)

        # Normalize depth map for display
        depth_visual = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_visual = depth_visual.astype(np.uint8)

        # Show depth map in real-time
        cv2.imshow("Depth Map", depth_visual)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()