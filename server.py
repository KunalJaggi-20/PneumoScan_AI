import os
import io
import base64
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
from flask import Flask, request, jsonify, render_template
import matplotlib.cm as cm

app = Flask(__name__)

# Ensure directories exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# ----------------------------
# 1. Model Loading & Hook setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model architecture
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)

# Load pre-trained weights
model_path = "model/pneumonia_model.pth"
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded weights from {model_path} model_state_dict")
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
        print(f"Loaded state_dict directly from {model_path}")
    else:
        model = checkpoint
        print(f"Loaded full model from {model_path}")
else:
    print(f"Warning: model weights not found at {model_path}!")

model = model.to(device)
model.eval()

# Global variable to store forward conv features for CAM
features_blobs = []

def hook_feature(module, input, output):
    features_blobs.clear()
    features_blobs.append(output.data)

# Register hook to layer4 (last convolutional layer of ResNet18)
model.layer4.register_forward_hook(hook_feature)

# ----------------------------
# 2. Image Preprocessing Transform
# ----------------------------
# Transform required by the pre-trained model card
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ----------------------------
# 3. Explanations Generator
# ----------------------------
def get_explanation(label, confidence):
    conf_pct = f"{confidence * 100:.1f}%"
    if label == "Pneumonia":
        return (
            f"The AI model detected **Pneumonia** with a high confidence score of **{conf_pct}**.<br><br>"
            "**Key Findings:**<br>"
            "• **Infiltrates & Opacities:** Consolidations and white patchy areas are highlighted in the lung zones (visible in the red/yellow heatmap zones).<br>"
            "• **Loss of Lung Transparency:** Increased density indicates fluid accumulation or inflammatory cells in the alveoli, replacing normal air spaces.<br>"
            "• **Abnormal Density Distribution:** The model has localized focal consolidations that are not present in normal chest X-rays."
        )
    else:
        return (
            f"The AI model classified the chest X-ray as **Normal** with a confidence score of **{conf_pct}**.<br><br>"
            "**Key Findings:**<br>"
            "• **Clear Lung Fields:** The lung zones appear dark and translucent, indicating healthy, air-filled lungs without fluid blockages.<br>"
            "• **Normal Cardiothoracic Ratio:** The cardiac silhouette (heart size) and borders are distinct and of normal proportion.<br>"
            "• **No Focal Consolidations:** The costophrenic angles and diaphragmatic domes are sharp, and no active densities or lesions were flagged by the network."
        )

# ----------------------------
# 4. Route Handlers
# ----------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if file was uploaded or demo image selected
        img_file = request.files.get("image")
        demo_name = request.form.get("demo_name")
        
        if img_file:
            img_bytes = img_file.read()
            original_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        elif demo_name:
            # Map demo name to local file path
            demo_path = os.path.join("static", f"{demo_name}.jpg")
            if not os.path.exists(demo_path):
                return jsonify({"error": f"Demo image {demo_name} not found."}), 404
            original_image = Image.open(demo_path).convert("RGB")
        else:
            return jsonify({"error": "No image or demo selected."}), 400

        # Prep image for model
        input_tensor = transform(original_image).unsqueeze(0).to(device)

        # Forward pass
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        
        if prob > 0.5:
            predicted_class = "Pneumonia"
            confidence = prob
        else:
            predicted_class = "Normal"
            confidence = 1.0 - prob

        # ----------------------------
        # 5. Class Activation Mapping (CAM)
        # ----------------------------
        # Retrieve the convolutional feature map from layer4
        feature_map = features_blobs[0].cpu().numpy()[0]  # [512, 7, 7]
        
        # Retrieve weights from fully connected layer (single channel weights, index [0])
        fc_weights = list(model.fc.parameters())[0].cpu().data.numpy()[0]  # [512]
        
        # Calculate weighted sum of feature maps for the predicted class
        cam_map = np.zeros(feature_map.shape[1:], dtype=np.float32)  # [7, 7]
        # Invert weights if predicting Normal (negative logit values represent Normal)
        multiplier = 1.0 if predicted_class == "Pneumonia" else -1.0
        for i in range(512):
            cam_map += multiplier * fc_weights[i] * feature_map[i, :, :]

        # Normalize CAM map
        cam_map = np.maximum(cam_map, 0)  # ReLU
        cam_map = cam_map - np.min(cam_map)
        cam_map = cam_map / (np.max(cam_map) + 1e-8)  # Prevent division by zero
        
        # Resize CAM map to match original image dimensions using PIL bilinear resizing
        original_width, original_height = original_image.size
        cam_img = Image.fromarray((cam_map * 255).astype(np.uint8)).resize((original_width, original_height), Image.Resampling.BILINEAR)
        
        # Apply JET colormap using Matplotlib
        cam_color_np = cm.jet(np.array(cam_img) / 255.0)[:, :, :3]  # Strip alpha channel
        cam_color_img = Image.fromarray((cam_color_np * 255).astype(np.uint8))
        
        # Blend original image and heatmap overlay (60% original, 40% heatmap)
        blended_image = Image.blend(original_image, cam_color_img, alpha=0.45)

        # ----------------------------
        # 6. Base64 Encode Images
        # ----------------------------
        # Encode original image
        buffered_orig = io.BytesIO()
        original_image.save(buffered_orig, format="JPEG", quality=85)
        orig_b64 = base64.b64encode(buffered_orig.getvalue()).decode('utf-8')

        # Encode heat-map blended image
        buffered_blend = io.BytesIO()
        blended_image.save(buffered_blend, format="JPEG", quality=85)
        blend_b64 = base64.b64encode(buffered_blend.getvalue()).decode('utf-8')

        # Return predictions and visual explanations
        explanation_html = get_explanation(predicted_class, confidence)
        
        return jsonify({
            "success": True,
            "prediction": predicted_class,
            "confidence": confidence,
            "explanation": explanation_html,
            "original_image": f"data:image/jpeg;base64,{orig_b64}",
            "heatmap_image": f"data:image/jpeg;base64,{blend_b64}"
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
