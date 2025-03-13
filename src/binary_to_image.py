import os
import numpy as np
from PIL import Image
import struct

# Function to recover images from binary file
def recover_images_from_binary(binary_file, output_dir):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the binary file for reading
    with open(binary_file, 'rb') as f:
        # Read the header
        magic_number = struct.unpack('>I', f.read(4))[0]
        if magic_number != 2051:
            raise ValueError(f"Invalid magic number {magic_number}. Not an image file.")

        num_images = struct.unpack('>I', f.read(4))[0]
        img_height = struct.unpack('>I', f.read(4))[0]
        img_width = struct.unpack('>I', f.read(4))[0]

        print(f"Recovering {num_images} images of size {img_height}x{img_width} from {binary_file}")

        # Read the image data
        for i in range(num_images):
            img_data = f.read(img_height * img_width)  # Read each image (as a flat array)
            img_array = np.frombuffer(img_data, dtype=np.uint8).reshape((img_height, img_width))

            # Convert the array back to an image
            img = Image.fromarray(img_array, 'L')  # 'L' mode for grayscale

            # Save the image to the output directory
            img.save(os.path.join(output_dir, f'image_{i}.png'))
            print(f"Saved image_{i}.png")

# Example usage
binary_file = '../archive/refactored_trace/input/mnist_test'  # Path to the binary file
output_directory = 'recovered_images'  # Directory to save the recovered images
recover_images_from_binary(binary_file, output_directory)
print(f"Recovered images saved to {output_directory}")
