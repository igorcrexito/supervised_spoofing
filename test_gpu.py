import tensorflow as tf

# Print available physical devices
print("Available devices:")
for device in tf.config.list_physical_devices():
    print(device)

# Check if a GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("\n✅ TensorFlow is using the GPU.")
    for gpu in gpus:
        print(f"  - {gpu}")
else:
    print("\n❌ No GPU detected. TensorFlow is using the CPU.")
