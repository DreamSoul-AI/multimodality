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

# Function to save images to binary file
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

if __name__ == "__main__":
    action = input("Do you want to 'recover' images from a binary file or 'save' images to a binary file? (recover/save): ").strip().lower()

    if action == "recover":
        binary_file = input("Enter the path to the binary file: ").strip()
        output_directory = input("Enter the directory to save recovered images: ").strip()
        recover_images_from_binary(binary_file, output_directory)
        print(f"Recovered images saved to {output_directory}")

    elif action == "save":
        image_directory = input("Enter the directory containing images to save as binary: ").strip()
        output_binary_file = input("Enter the output path for the binary file: ").strip()
        images = load_images_from_directory(image_directory)
        save_images_to_binary(images, output_binary_file)
        print(f"Saved {len(images)} images to {output_binary_file}")

    else:
        print("Invalid option. Please choose either 'recover' or 'save'.")
