import os


def get_input_size(directory):
    total_size = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg'):
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
    return total_size


def get_output_size(directory):
    total_size = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.nncp'):
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
    return total_size


if __name__ == "__main__":
    input_size_bytes = get_input_size("C:\\Users\\yuboh\\GitHub\\data\\mnist_0_to_19")
    # input_size_mb = total_size_bytes / (1024 * 1024)  # Convert bytes to MB
    output_size_bytes = get_output_size("files_compressed")
    ratio = input_size_bytes / output_size_bytes
    print(f"Compression Ratio: {ratio}")
