from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Load the trained model and tokenizer
model_path = r"D:\AIT_lecture\NLP\code\NLP-A5\dpo_lr5e-05_bs4_ep3_beta0.1\checkpoint-4"  # Make sure the model is saved in this path
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Ensure model is in evaluation mode
model.eval()

# Function to generate response
def generate_response(prompt, max_tokens=100):
    try:
        # Format input as a dialogue
        formatted_prompt = f"Human: {prompt}\n\nAssistant:"

        # Tokenize the input
        input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids

        # Generate a response
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_tokens,  # Controls output length
                temperature=0.7,  # Adds diversity
                top_p=0.9,  # Nucleus sampling
                do_sample=True,  # Enables varied responses
                pad_token_id=tokenizer.eos_token_id,  # Handles padding properly
            )

        # Decode and clean response
        full_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = full_response.replace(formatted_prompt, "").strip()

        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form["user_input"]
        response = generate_response(user_input)
        return render_template("index.html", user_input=user_input, response=response)
    return render_template("index.html", user_input=None, response=None)

if __name__ == "__main__":
    app.run(debug=True)
