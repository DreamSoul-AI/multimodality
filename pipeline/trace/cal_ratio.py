import os


def get_input_size(directory):
    total_size = 0
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            total_size += os.path.getsize(file_path)
    return total_size


def get_output_size(directory):
    total_size = 0
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            total_size += os.path.getsize(file_path)
    return total_size


if __name__ == "__main__":
    # input_size_bytes = get_input_size("input/dickens_test_text")
    input_size_bytes = os.path.getsize('input/dickens_test_text')

    print("input_size_bytes: {}".format(input_size_bytes/(1024*1024)))
    # input_size_mb = total_size_bytes / (1024 * 1024)  # Convert bytes to MB
    # output_size_bytes = get_output_size("output")
    output_file_path = "dickens_test_64_256_4096_bs8_1_seq8.compressed.combined"
    output_size_bytes = os.path.getsize(output_file_path)
    print("output_size_bytes: {}".format(output_size_bytes/(1024*1024)))
    ratio = input_size_bytes / output_size_bytes
    print(f"Compression Ratio: {ratio}")