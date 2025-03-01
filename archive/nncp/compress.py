import os
import subprocess

# Define the input and output directories
input_dir = r"C:\Users\yuboh\GitHub\data\mnist_0_to_19"
output_dir = "files_compressed"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Loop through each file in the input directory
for file in os.listdir(input_dir):
    file_path = os.path.join(input_dir, file)

    # Check if it's a file
    if os.path.isfile(file_path):
        # Extract the base filename without the extension
        base = os.path.splitext(file)[0]
        # Set the output path with the .nncp extension
        output_path = os.path.join(output_dir, f"{base}.nncp")

        # Run the nncp command with the current file
        subprocess.run(["nncp-2024-06-05-win64/nncp.exe", "c", file_path, output_path])
