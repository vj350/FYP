import scipy.io as sio
from pathlib import Path

file_path = "data/B01T.mat"
mat = sio.loadmat(file_path)

data = mat["data"]

block = data[0, 0]          # first of the 3 blocks
block = block[0, 0]         # unwrap structured array

print("Fields in one block:")
print(block.dtype.names)

for field in block.dtype.names:
    value = block[field]
    print(f"\nField: {field}")
    print("Type:", type(value))
    try:
        print("Shape:", value.shape)
        print("Dtype:", value.dtype)
    except AttributeError:
        print("Value:", value)

# Now inspect useful fields more deeply
X = block["X"]
trial = block["trial"]
y = block["y"]
fs = block["fs"]
artifacts = block["artifacts"]

print("\n--- Deeper inspection ---")
print("X shape:", X.shape)
print("trial shape:", trial.shape)
print("y shape:", y.shape)
print("fs:", fs)
print("artifacts shape:", artifacts.shape)

print("\nFirst few trial indices:")
print(trial.flatten()[:10])

print("\nFirst few labels:")
print(y.flatten()[:10])

print("\nFirst few artifact flags:")
print(artifacts.flatten()[:10])