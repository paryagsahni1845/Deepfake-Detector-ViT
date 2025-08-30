from flask import Flask, request, jsonify, render_template
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

app = Flask(__name__)

# Load model from HF Hub (your repo)
model = ViTForImageClassification.from_pretrained("Sxhni/deepfake-detector-vit")
processor = ViTImageProcessor.from_pretrained("Sxhni/deepfake-detector-vit")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = Image.open(file.stream)

    # Preprocess & forward pass
    inputs = processor(images=img, return_tensors="pt")
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0].detach().numpy()

    # Find predicted label + confidence
    pred_id = probs.argmax()
    pred_label = model.config.id2label[pred_id]
    confidence = float(probs[pred_id])

    # Full distribution
    probabilities = {model.config.id2label[i]: float(probs[i]) for i in range(len(probs))}

    return jsonify({
        "prediction": pred_label,
        "confidence": confidence
    })

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
