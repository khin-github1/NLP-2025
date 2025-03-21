# app.py
from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

# Load your trained model (change path if you saved under a different name)
model = AutoModelForSequenceClassification.from_pretrained("./odd_student_model")
tokenizer = AutoTokenizer.from_pretrained("./odd_student_model")
model.eval()

# Inference function
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return "Toxic" if pred == 1 else "Non-Toxic"

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    text_input = ""
    if request.method == "POST":
        text_input = request.form["text"]
        result = classify_text(text_input)
    return render_template("index.html", result=result, text_input=text_input)

if __name__ == "__main__":
    app.run(debug=True)
