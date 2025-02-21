import numpy as np

def compare_files(original_file_path, decoded_file_path):
    # Read the original file
    with open(original_file_path, 'rb') as fp:
        original_data = np.frombuffer(fp.read(), dtype=np.uint8)

    # Read the decoded file
    with open(decoded_file_path, 'rb') as fp:
        decoded_data = np.frombuffer(fp.read(), dtype=np.uint8)

    # Compare both arrays
    if np.array_equal(original_data, decoded_data):
        print("The decoded file matches the original file.")
    else:
        print("The decoded file does NOT match the original file.")

compare_files('input/dickens_test', 'decompressed')
