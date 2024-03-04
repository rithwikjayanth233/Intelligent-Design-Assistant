import PyPDF2
import nltk
import os

# Download necessary NLTK resources
nltk.download('punkt')

pdf_path = '/home/rithwik/paper/dataset/shigley.pdf'  # Replace with your PDF path
folder_path = '/home/rithwik/paper/dataset'  # Replace with your desired folder

# Load stop words
stop_words = nltk.corpus.stopwords.words('english')

def extract_and_preprocess_text(pdf_path):
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()

        text = text.lower()
        # Tokenize the text (returning a list)
        text = nltk.word_tokenize(text)
        return text
    except FileNotFoundError:
        print(f"Error: PDF file not found at '{pdf_path}'")
        return None  # Return None to indicate an error
    except PermissionError:
        print(f"Error: Insufficient permissions to access the PDF file")
        return None

def remove_stop_words(text):
    """
    Removes stop words from the list of words.

    Args:
        text: A list of words (output from extract_and_preprocess).

    Returns:
        A list of words with stop words removed, or None if error occurred.
    """
    if text is None:
        return None  # Propagate error

    # Remove stop words from the list of words
    filtered_words = [word for word in text if word not in stop_words]
    return filtered_words

text = extract_and_preprocess_text(pdf_path)
if text is not None:
    # Join the list elements into a string (fix for the AttributeError)
    cleaned_text = " ".join(remove_stop_words(text))
    if cleaned_text is not None:
        cleaned_text_file_path = os.path.join(folder_path, "shigley_cleaned_text.txt")
        with open(cleaned_text_file_path, "w") as text_file:
            text_file.write(cleaned_text)
        print("Text cleaning and saving completed!")
else:
    print("Error occurred during processing. Please check the logs.")
