import numpy as np


# def compare_files(original_file_path, decoded_file_path):
#     # Read the original file
#     with open(original_file_path, 'rb') as fp:
#         original_data = np.frombuffer(fp.read(), dtype=np.uint8)
#
#     # Read the decoded file
#     with open(decoded_file_path, 'rb') as fp:
#         decoded_data = np.frombuffer(fp.read(), dtype=np.uint8)
#
#     # Compare both arrays
#     if np.array_equal(original_data, decoded_data):
#         print("Same!")
#     else:
#         print("Not same!")


def compare_files(original_file_path, decoded_file_path):
    # Read the original file
    with open(original_file_path, 'rb') as fp:
        original_data = np.frombuffer(fp.read(), dtype=np.uint8)

    # Read the decoded file
    with open(decoded_file_path, 'rb') as fp:
        decoded_data = np.frombuffer(fp.read(), dtype=np.uint8)

    # Compare both arrays
    if original_data.shape != decoded_data.shape:
        print(f"Files are of different sizes. Original: {original_data.shape[0]} bytes, Decoded: {decoded_data.shape[0]} bytes")
        return

    # Find differences
    mismatches = np.where(original_data != decoded_data)[0]

    if len(mismatches) == 0:
        print("Same!")
    else:
        print(f"Not same! Found {len(mismatches)} differences.")
        print("Mismatched byte positions and values (up to first 10 differences):")
        for i in range(min(100, len(mismatches))):
            index = mismatches[i]
            print(f"Position {index}: Original byte = {original_data[index]}, Decoded byte = {decoded_data[index]}")

compare_files(r'C:\Users\yuboh\GitHub\multimodality\src\data\DICKENS\raw\dickens',
              r'C:\Users\yuboh\GitHub\multimodality\src\output\result\0_DICKENS_bootstrap\output.decompressed')
