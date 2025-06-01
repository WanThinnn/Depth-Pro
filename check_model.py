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
print("Model and transforms loaded successfully.")
print(f"Using device: {device}")
