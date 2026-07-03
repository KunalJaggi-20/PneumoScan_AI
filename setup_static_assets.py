import os
import shutil
import urllib.request

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(base_dir, "static")
    templates_dir = os.path.join(base_dir, "templates")
    
    # Create directories
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(templates_dir, exist_ok=True)
    print("Created static/ and templates/ directories.")
    
    # Copy pneumonia demo image
    src_pneu = "c:/Users/Kunal Jaggi/Downloads/Pneumonia_x_ray.jpg"
    dest_pneu = os.path.join(static_dir, "demo_pneumonia.jpg")
    
    if os.path.exists(src_pneu):
        shutil.copy(src_pneu, dest_pneu)
        print(f"Copied local user X-ray to {dest_pneu}")
    else:
        print(f"Warning: local user X-ray not found at {src_pneu}")
        
    # Download normal demo image
    normal_url = "https://github.com/bentoml/Pneumonia-Detection-Demo/raw/main/samples/NORMAL2-IM-1427-0001.jpeg"
    dest_normal = os.path.join(static_dir, "demo_normal.jpg")
    
    print(f"Downloading normal demo image from: {normal_url}")
    try:
        req = urllib.request.Request(normal_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(dest_normal, "wb") as out_file:
            out_file.write(response.read())
        print(f"Successfully downloaded and saved normal demo image to {dest_normal}")
    except Exception as e:
        print(f"Error downloading normal image: {e}")

if __name__ == "__main__":
    main()
