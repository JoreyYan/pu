import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

# from models.flow_model import SideAtomsFlowModel
# from omegaconf import OmegaConf
#
# print("Loading config...")
# cfg = OmegaConf.load('configs/your_config.yaml')
#
# print("Creating model...")
# model = SideAtomsFlowModel(cfg.model)
#
# print("✓ Model created successfully!")
# print("Moving to GPU...")
# model = model.cuda()
# print("✓ All tests passed!")
