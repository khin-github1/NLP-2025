# app.py
from flask import Flask, render_template, request
import torch
from utils import load_vocab, load_model, translate_text

app = Flask(__name__)

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vocabulary and model (using the parameters from your file)
vocab_transform = load_vocab()  # expects file at app/models/vocabs.pkl
model = load_model(device=device, vocab_transform=vocab_transform)  # expects model at app/models/add_model_v4.pt

@app.route('/', methods=['GET', 'POST'])
def index():
    translation = ""
    input_text = ""
    if request.method == 'POST':
        input_text = request.form.get('english_text', '')
        if input_text.strip():
            translation = translate_text(input_text, model, vocab_transform, device)
    return render_template('index.html', translation=translation, input_text=input_text)

if __name__ == '__main__':
    app.run(debug=True)