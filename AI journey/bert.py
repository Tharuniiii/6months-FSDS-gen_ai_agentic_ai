import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from fastapi import FastAPI
from pydantic import BaseModel
import streamlit as st
import requests

# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Load the IMDB dataset
dataset = load_dataset("imdb")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

# Apply the tokenization function to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # Output directory
    evaluation_strategy="epoch",     # Evaluate every epoch
    learning_rate=2e-5,              # Learning rate for optimization
    per_device_train_batch_size=8,   # Batch size for training
    per_device_eval_batch_size=8,    # Batch size for evaluation
    num_train_epochs=3,              # Number of epochs to train
    weight_decay=0.01,               # Strength of weight decay (regularization)
    logging_dir='./logs',            # Directory for storing logs
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                         # The model to train
    args=training_args,                  # Training arguments
    train_dataset=tokenized_datasets['train'],  # Training dataset
    eval_dataset=tokenized_datasets['test'],   # Evaluation dataset
)

# Start the training process
trainer.train()

# After fine-tuning, use the model for predictions
inputs = tokenizer(["I love this product!", "This is terrible, do not buy it."], padding=True, truncation=True, return_tensors="pt")

# Ensure model is in evaluation mode
model.eval()

# Move inputs and model to the same device (e.g., CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move the model to the device
inputs = inputs.to(device)  # Move the inputs to the device

# Get predictions
with torch.no_grad():  # Use torch.no_grad() for inference
    outputs = model(**inputs)
    logits = outputs.logits

# Convert logits to probabilities
probs = torch.nn.functional.softmax(logits, dim=-1)  # Use softmax to get probabilities

# Get predicted class (0 or 1 for binary classification)
predictions = torch.argmax(probs, dim=-1)  # Use argmax to get the predicted class
print(predictions)

# Evaluate the model on the test set
eval_results = trainer.evaluate()
print(eval_results)

# Save the trained model
model.save_pretrained('./bert_imdb_model')
tokenizer.save_pretrained('./bert_imdb_model')

# Load the trained model and tokenizer for future predictions
model = BertForSequenceClassification.from_pretrained('./bert_imdb_model')
tokenizer = BertTokenizer.from_pretrained('./bert_imdb_model')

# Function to make predictions
def predict(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():  # Use no_grad for inference
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(axis=-1)
    return predictions

# FastAPI application setup
app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.post("/predict/")
def predict_api(request: TextRequest):
    inputs = tokenizer(request.text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():  # Use no_grad for inference
        outputs = model(**inputs)
        prediction = outputs.logits.argmax(axis=-1).item()
    return {"prediction": prediction}

# Streamlit frontend
def run_streamlit():
    st.title("Sentiment Analysis using BERT")
    
    # User input for text to analyze
    user_input = st.text_area("Enter text for sentiment analysis:", "I love this product!")

    if st.button("Analyze Sentiment"):
        # Call the FastAPI backend for prediction
        response = requests.post("http://127.0.0.1:8000/predict/", json={"text": user_input})
        
        if response.status_code == 200:
            prediction = response.json()["prediction"]
            sentiment = "Positive" if prediction == 1 else "Negative"
            st.write(f"The sentiment of the text is: {sentiment}")
        else:
            st.write("Error in prediction")

if __name__ == "__main__":
    # Run FastAPI app
    import uvicorn
    from threading import Thread

    def run_api():
        uvicorn.run(app, host="127.0.0.1", port=8000)

    # Start the FastAPI app in a separate thread
    thread = Thread(target=run_api)
    thread.start()

    # Start the Streamlit frontend
    run_streamlit()
