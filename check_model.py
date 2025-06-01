from PIL import Image
import torch
import depth_pro
import os
import numpy as np

# Hàm để lấy device GPU nếu có
def get_torch_device() -> torch.device:
    """Get the Torch device."""
    device = torch.device("cuda:0")
    # if torch.cuda.is_available():
    #     device = torch.device("cuda:0")
    # elif torch.backends.mps.is_available():  # Cho MacOS với Apple Silicon
    #     device = torch.device("mps")
    return device

# Load model và preprocessing transform với GPU
device = get_torch_device()
model, transform = depth_pro.create_model_and_transforms(
    device=device,
    precision=torch.half  # Sử dụng half precision để tiết kiệm VRAM
)
model.eval()

print(f"Using device: {device}")

image_path = "./img/portrait-1.png"
# Load and preprocess an image.
image, _, f_px = depth_pro.load_rgb(image_path)
image = transform(image)

# Run inference.
prediction = model.infer(image, f_px=f_px)
depth = prediction["depth"]  # Depth in [m].
focallength_px = prediction["focallength_px"]  # Focal length in pixels.

# Lưu theo format depth-pro
import depth_pro.utils as utils

output_path = "output"
os.makedirs(output_path, exist_ok=True)

# Lưu depth map
depth_output = os.path.join(output_path, "depth.npy")
np.save(depth_output, depth.cpu().numpy())

# Lưu focal length
focal_output = os.path.join(output_path, "focal_length.txt")
with open(focal_output, "w") as f:
    f.write(f"{focallength_px}")

print(f"Results saved to {output_path}/")