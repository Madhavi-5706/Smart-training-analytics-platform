import tensorflow as tf
import torch
import sys

print("=== GPU Configuration and Diagnostics ===\n")

print("Python Version:", sys.version)

print("\nPyTorch GPU Status:")
print("PyTorch Version:", torch.__version__)
print("PyTorch CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("PyTorch CUDA Device:", torch.cuda.get_device_name(0))

print("\nTensorFlow GPU Status:")
print("TensorFlow Version:", tf.__version__)
print("\nAll Physical Devices:", tf.config.list_physical_devices())
print("\nGPU Devices:", tf.config.list_physical_devices('GPU'))

if tf.config.list_physical_devices('GPU'):
    print("\nTensorFlow GPU devices found")
else:
    print("\nNo TensorFlow GPU devices found")

print("Available devices:", tf.config.list_physical_devices())

print("\n=== End GPU Configuration and Diagnostics ===")
