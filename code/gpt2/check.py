from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import nltk
import PyPDF2
import os

# Download necessary NLTK resources
nltk.download('punkt')

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load the PDF and preprocess the text
pdf_path = '/home/rithwik/paper/dataset/shigley.pdf'  # Replace with your PDF path

try:
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

    text = text.lower()
    # Tokenize the text
    text = nltk.word_tokenize(text)
except FileNotFoundError:
    print(f"Error: PDF file not found at '{pdf_path}'")
    text = None
except PermissionError:
    print(f"Error: Insufficient permissions to access the PDF file")
    text = None

# Check if text is extracted successfully
if text:
    # Initialize GPT-2 model and config
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    config = GPT2Config.from_pretrained("gpt2")

    # Further processing and training...
else:
    print("Error occurred during text extraction. Please check the logs.")
