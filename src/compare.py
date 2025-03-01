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
        print("Same!")
    else:
        print("Not same!")

compare_files(r'C:\Users\yuboh\GitHub\multimodality\src\output\dict\DICKENS\params_dickens', r'C:\Users\yuboh\GitHub\Dzip-torch\coding-gpu\params_dickens')
