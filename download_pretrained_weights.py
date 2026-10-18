import os
import urllib.request
import torch

MODEL_URL = "https://huggingface.co/luanacarolina/pneumonia-chest-xray-classifier/resolve/main/checkpoints/binary/densenet121.pt"

def download_weights(output_path=None):
    if output_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        output_path = os.path.join(model_dir, "pneumonia_model.pth")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"Downloading refined DenseNet121 weights from Hugging Face...")
    print(f"URL: {MODEL_URL}")
    print(f"Destination: {output_path}")

    temp_path = output_path + ".tmp"
    try:
        req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(temp_path, "wb") as out_file:
            out_file.write(resp.read())

        # Load and clean state_dict (stripping optimizer states to reduce size from 84MB to ~27MB)
        checkpoint = torch.load(temp_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            state_dict = checkpoint.state_dict()

        torch.save(state_dict, output_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Optimized model saved successfully! Size: {size_mb:.2f} MB")
        return True
    except Exception as e:
        print(f"Error downloading or processing weights: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

def main():
    download_weights()

if __name__ == "__main__":
    main()

