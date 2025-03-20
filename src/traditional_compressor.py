import os
import subprocess
import shutil
import gzip
import zstandard as zstd


def get_file_size(file_path):
    return os.path.getsize(file_path)


def compress_with_7zip(input_file, output_file):
    try:
        subprocess.run(['7z', 'a', output_file, input_file], check=True)
    except FileNotFoundError:
        print("7z command not found. Make sure 7-Zip is installed and accessible from the command line.")
        return


def compress_with_gzip(input_file, output_file):
    with open(input_file, 'rb') as f_in, gzip.open(output_file, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)


def compress_with_zstd(input_file, output_file, compression_level=19):
    with open(input_file, 'rb') as f_in, open(output_file, 'wb') as f_out:
        compressor = zstd.ZstdCompressor(level=compression_level)
        compressor.copy_stream(f_in, f_out)


def run_experiment(input_file):
    original_size = get_file_size(input_file)

    output_dir = "./output/traditional_compressed"
    os.makedirs(output_dir, exist_ok=True)

    # 7zip compression
    compressed_7z_file = os.path.join(output_dir, os.path.basename(input_file) + '.7z')
    compress_with_7zip(input_file, compressed_7z_file)
    compressed_7z_size = get_file_size(compressed_7z_file)
    compression_ratio_7z = original_size / compressed_7z_size
    print(f"7zip Compression Ratio: {compression_ratio_7z}")

    # Gzip compression
    compressed_gzip_file = os.path.join(output_dir, os.path.basename(input_file) + '.gz')
    compress_with_gzip(input_file, compressed_gzip_file)
    compressed_gzip_size = get_file_size(compressed_gzip_file)
    compression_ratio_gzip = original_size / compressed_gzip_size
    print(f"Gzip Compression Ratio: {compression_ratio_gzip}")

    # Zstd compression
    compressed_zstd_file = os.path.join(output_dir, os.path.basename(input_file) + '.zst')
    compress_with_zstd(input_file, compressed_zstd_file)
    compressed_zstd_size = get_file_size(compressed_zstd_file)
    compression_ratio_zstd = original_size / compressed_zstd_size
    print(f"Zstd Compression Ratio: {compression_ratio_zstd}")

    return compression_ratio_7z, compression_ratio_gzip, compression_ratio_zstd


if __name__ == "__main__":
    input_file = "data/DICKENS/raw/dickens"  # Replace with the path to your dataset
    run_experiment(input_file)
