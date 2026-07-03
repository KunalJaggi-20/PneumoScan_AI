import os
import urllib.request

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    
    url = "https://huggingface.co/izeeek/resnet18_pneumonia_classifier/resolve/main/resnet18_pneumonia_classifier.pth"
    output_path = os.path.join(model_dir, "pneumonia_model.pth")
    
    print(f"Downloading pre-trained weights from Hugging Face...")
    print(f"URL: {url}")
    print(f"Saving to: {output_path}")
    
    try:
        # Download the file
        urllib.request.urlretrieve(url, output_path)
        print("Download completed successfully!")
        
        # Verify size
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Model size: {size_mb:.2f} MB")
        
    except Exception as e:
        print(f"Error downloading weights: {e}")

if __name__ == "__main__":
    main()
