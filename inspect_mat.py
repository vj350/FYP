import scipy.io as sio
from pathlib import Path

file_path = "data/B01T.mat"

mat = sio.loadmat(file_path)

print("Top-level keys:")
for k in mat.keys():
    if not k.startswith("__"):
        print(k, type(mat[k]))

data = mat["data"]
print("\nType of data:", type(data))
print("Shape of data:", data.shape)
print("Dtype of data:", data.dtype)

# If it is a cell array, inspect each element
if data.dtype == object:
    for i in range(data.shape[1] if data.ndim > 1 else data.shape[0]):
        elem = data[0, i] if data.ndim > 1 else data[i]
        print(f"\nElement {i}:")
        print("  type:", type(elem))
        try:
            print("  shape:", elem.shape)
            print("  dtype:", elem.dtype)
        except AttributeError:
            print("  value:", elem)