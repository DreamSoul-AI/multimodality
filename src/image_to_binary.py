import os
import numpy as np
from PIL import Image
import struct


def save_images_to_binary(images, output_file):
    # Get image dimensions (assumes all images are the same size)
    num_images = len(images)
    img_height, img_width = images[0].shape

    # Open binary file for writing
    with open(output_file, 'wb') as f:
        # Write the header
        # Magic number (2051 is used in MNIST for image files)
        f.write(struct.pack('>I', 2051))  # '>I' for big-endian unsigned integer
        f.write(struct.pack('>I', num_images))  # Number of images
        f.write(struct.pack('>I', img_height))  # Number of rows (height)
        f.write(struct.pack('>I', img_width))  # Number of columns (width)

        # Write the image data (pixel values)
        for img in images:
            img = img.astype(np.uint8)  # Ensure the image is in uint8 format (0-255)
            img.tobytes()  # Convert image to bytes
            f.write(img.tobytes())  # Write the pixel data to the file


# Function to load images from a directory
def load_images_from_directory(image_dir, img_size=(28, 28)):
    images = []
    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img = img.resize(img_size)  # Resize to (28, 28) or desired size
        img_array = np.array(img)  # Convert to NumPy array
        images.append(img_array)
    return images


# Example usage
image_directory = r'C:\Users\yuboh\GitHub\data\mnist_test'
images = load_images_from_directory(image_directory)

# Save the images to a binary file
output_binary_file = '../archive/refactored_trace/input/mnist_test'
save_images_to_binary(images, output_binary_file)
print(f"Saved {len(images)} images to {output_binary_file}")
