import torch
import os

# Exact location of the .pt file
file_path = r"C:\Users\Zahoor Ahmad`\OneDrive\Desktop\Confrence Paper first 1\ftnct_project\attack_segment_blackbox.pt"

# Check whether the file exists
print("File exists:", os.path.exists(file_path))
print("Loading:", file_path)

# Load the PyTorch file
data = torch.load(file_path, map_location="cpu")

# Show the type of data
print("\nData type:")
print(type(data))

print("\n" + "=" * 60)

# If the file contains a dictionary
if isinstance(data, dict):
    print("The file contains a dictionary.")
    print("\nKeys inside the file:")
    print(list(data.keys()))

    print("\nContents:")
    for key, value in data.items():
        print(f"\nKey: {key}")
        print("Type:", type(value))

        if isinstance(value, torch.Tensor):
            print("Tensor shape:", value.shape)
            print("Data type:", value.dtype)
            print("First 10 values:")
            print(value.flatten()[:10])
        else:
            print("Value:", value)

# If the file directly contains a tensor
elif isinstance(data, torch.Tensor):
    print("The file contains a PyTorch Tensor.")
    print("Shape:", data.shape)
    print("Data type:", data.dtype)
    print("\nFirst 10 values:")
    print(data.flatten()[:10])

else:
    print("Contents:")
    print(data)

print("\n" + "=" * 60)
print("DONE")