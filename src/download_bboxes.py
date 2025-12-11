import os
import requests

BASE_URL = "https://raw.githubusercontent.com/Aminoid/The-Nature-Conservancy-Fisheries-Monitoring/master/BBFish/annos/"
# List of files identified from the GitHub API response earlier
FILES = [
    "alb_labels.json",
    "bet_labels.json",
    "dol_labels.json",
    "lag_labels.json",
    "other_labels.json",
    "shark_labels.json",
    "yft_labels.json"
]

OUTPUT_DIR = "data/bounding_boxes"

def download_file(filename):
    url = BASE_URL + filename
    output_path = os.path.join(OUTPUT_DIR, filename)
    print(f"Downloading {filename}...")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Saved to {output_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {filename}: {e}")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    for filename in FILES:
        download_file(filename)
    
    print("Download complete.")
