import urllib.request
import sys
import time

url = "https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp312-cp312-win_amd64.whl"
filename = "torch-2.5.1+cu121-cp312-cp312-win_amd64.whl"

start_time = time.time()
last_print_time = start_time

def reporthook(count, block_size, total_size):
    global last_print_time
    current_time = time.time()
    
    # Print every 5 seconds
    if current_time - last_print_time >= 5.0 or count == 0:
        downloaded = count * block_size
        progress = downloaded / total_size * 100 if total_size > 0 else 0
        speed = downloaded / (current_time - start_time) / (1024 * 1024) if current_time > start_time else 0
        
        print(f"Downloaded {downloaded / (1024 * 1024):.2f} MB / {total_size / (1024 * 1024):.2f} MB ({progress:.2f}%) at {speed:.2f} MB/s", flush=True)
        last_print_time = current_time

print(f"Starting download of {url}...")
urllib.request.urlretrieve(url, filename, reporthook)
print("Download complete!")
