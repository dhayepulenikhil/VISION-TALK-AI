from flask import Flask, request, jsonify, render_template
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import torch
import io

app = Flask(__name__)

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD MODEL ----------------
processor = ViltProcessor.from_pretrained(
    "dandelin/vilt-b32-finetuned-vqa"
)
model = ViltForQuestionAnswering.from_pretrained(
    "dandelin/vilt-b32-finetuned-vqa"
)
model.to(device)
model.eval()

# ---------------- GLOBAL IMAGE ----------------
uploaded_image = None

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")

# ---------- IMAGE UPLOAD ----------
@app.route("/upload", methods=["POST"])
def upload():
    global uploaded_image

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    image_bytes = file.read()
    uploaded_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    return jsonify({"status": "Image uploaded successfully"})

# ---------- QUESTION ASK ----------
@app.route("/ask", methods=["POST"])
def ask():
    global uploaded_image

    if uploaded_image is None:
        return jsonify({"answer": "Please upload an image first."})

    data = request.get_json()
    question = data.get("question", "").strip()
    language = data.get("language", "en")   # en / te / hi

    if question == "":
        return jsonify({"answer": "I did not hear your question clearly."})

    # -------- VILT PROCESS --------
    encoding = processor(
        images=uploaded_image,
        text=question,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoding)

    logits = outputs.logits
    idx = logits.argmax(-1).item()
    answer = model.config.id2label[idx]

    # -------- LANGUAGE PREFIX --------
    if language == "te":
        answer = "సమాధానం: " + answer
    elif language == "hi":
        answer = "उत्तर: " + answer

    return jsonify({"answer": answer})

# ---------------- MAIN ----------------
if __name__ == "__main__":
    app.run(debug=True)
