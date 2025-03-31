import cv2
import numpy as np
from PIL import Image
import pytesseract
import requests
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from urllib.parse import unquote, urlparse
from docx import Document

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class PlagiarismChecker:
    def __init__(self, threshold=0.5):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.threshold = threshold
    
    def preprocess_image(self, image_path):
        """
        Preprocess image to enhance OCR for handwritten text.
        """
        try:
            # Read image using OpenCV
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Apply thresholding to binarize the image
            _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Resize the image to enhance small handwriting
            resized_image = cv2.resize(binary_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

            # Apply GaussianBlur to reduce noise
            blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

            # Convert to PIL format for Tesseract
            preprocessed_image = Image.fromarray(blurred_image)
            return preprocessed_image
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None

    def extract_text_from_image(self, image_path):
        """
        Extract text from an image using Tesseract OCR with preprocessing.
        """
        try:
            # Preprocess image
            preprocessed_image = self.preprocess_image(image_path)

            if preprocessed_image is None:
                return ""

            # Configure Tesseract for handwritten text
            custom_config = r'--oem 1 --psm 3'

            # Perform OCR
            text = pytesseract.image_to_string(preprocessed_image, config=custom_config)
            return text.strip()
        except Exception as e:
            print(f"Error extracting text from {image_path}: {e}")
            return ""

    def preprocess_text(self, text):
        """
        Preprocess text by removing stopwords and converting to lowercase.
        """
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        return ' '.join(filtered_tokens)

    def calculate_similarity(self, text1, text2):
        """
        Calculate similarity between two texts using SequenceMatcher.
        """
        return SequenceMatcher(None, text1, text2).ratio()
    
    def search_internet(self, text, num_results=5):
        """
        Search the internet for similar content and return both links and their content.
        """
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            query = '+'.join(text.split())
            url = f"https://www.google.com/search?q={query}"
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")
            links = [
                a['href'] for a in soup.find_all('a', href=True) 
                if "url?q=" in a['href']
            ]
            
            web_contents = []
            for link in links[:num_results]:
                try:
                    clean_url = link.split("&")[0].replace("/url?q=", "")
                    clean_url = unquote(clean_url)
                    parsed_url = urlparse(clean_url)
                    
                    if parsed_url.scheme in ["http", "https"]:
                        page_response = requests.get(clean_url, headers=headers, timeout=5)
                        page_soup = BeautifulSoup(page_response.text, "html.parser")
                        content = page_soup.get_text()
                        web_contents.append((clean_url, content))
                except Exception as e:
                    print(f"Error fetching content from {link}: {e}")
            return web_contents
        except Exception as e:
            print(f"Error during web search: {e}")
            return []
        
    def calculate_plagiarism_percentage(self, source_text, web_content):
        """
        Calculate the percentage of the source text that is plagiarized based on matches in the web content.
        
        Args:
            source_text (str): Text to check for plagiarism
            web_content (str): Content fetched from a web page
        
        Returns:
            float: Percentage of the source text that is plagiarized
        """
        # Preprocess source and web content
        processed_source_text = self.preprocess_text(source_text)
        processed_web_content = self.preprocess_text(web_content)

        # Tokenize the preprocessed source and web content
        source_tokens = processed_source_text.split()
        web_tokens = processed_web_content.split()

        # Calculate the length of matching tokens
        matched_length = 0
        for token in source_tokens:
            if token in web_tokens:
                matched_length += len(token)  # Count the length of the matching token

        # Calculate the total length of the source text
        total_source_length = sum(len(token) for token in source_tokens)

        # Calculate plagiarism percentage
        plagiarized_percentage = (matched_length / total_source_length) if total_source_length > 0 else 0
        return plagiarized_percentage

    def check_plagiarism(self, input_text, num_results=10):
        """
        Check plagiarism for the given text by searching the internet.
        
        Args:
            input_text (str): Text to check for plagiarism
            num_results (int): Number of web pages to check
        
        Returns:
            dict: Plagiarism results with plagiarized percentages
        """
        print("Searching the internet for similar content...")
        web_results = self.search_internet(input_text, num_results=num_results)

        print("\nLinks Checked:")
        for i, (link, _) in enumerate(web_results, 1):
            print(f"Link {i}: {link}")
        
        plagiarism_results = {}
        for i, (link, web_content) in enumerate(web_results):
            # Calculate plagiarized percentage
            plagiarized_percentage = self.calculate_plagiarism_percentage(input_text, web_content)
            plagiarism_results[link] = {
                'plagiarized_percentage': plagiarized_percentage,
                'is_plagiarized': plagiarized_percentage >= self.threshold
            }
        
        return plagiarism_results

    def read_docx(self, file_path):
        """
        Read and extract text from a .docx file.
        
        Args:
            file_path (str): Path to the .docx file
        
        Returns:
            str: Extracted text from the .docx file
        """
        try:
            doc = Document(file_path)
            text = []
            for paragraph in doc.paragraphs:
                text.append(paragraph.text)
            return '\n'.join(text).strip()
        except Exception as e:
            print(f"Error reading .docx file: {e}")
            return ""

def main():
    # Initialize plagiarism checker
    checker = PlagiarismChecker(threshold=0.5)
    
    print("Choose input type for plagiarism check:")
    print("1. Image")
    print("2. Text")  
    print("3. Text File")
    print("4. Word Document (.docx)")
    choice = input("Enter your choice (1/2/3/4): ").strip()
    
    source_text = ""
    
    if choice == "1":
        # Image input
        source_image = input("Enter the path to the image: ").strip()
        source_text = checker.extract_text_from_image(source_image)
        print(f"\nExtracted Text from Image:\n{source_text}\n")
    elif choice == "2":
        # Text input
        source_text = input("Enter the text to check for plagiarism: ").strip()
    elif choice == "3":
        # Text file input
        text_file_path = input("Enter the tpath to the text file: ").strip()
        try:
            with open(text_file_path, 'r', encoding='utf-8') as file:
                source_text = file.read().strip()
            print(f"\nText from File:\n{source_text}\n")
        except Exception as e:
            print(f"Error reading text file: {e}")
            return
    elif choice == "4":
        # Word document (.docx) input
        docx_file_path = input("Enter the path to the .docx file: ").strip()
        source_text = checker.read_docx(docx_file_path)
        print(f"\nText from Word Document:\n{source_text}\n")
    else:
        print("Invalid choice. Exiting.")
        return
    
    # Preprocess the extracted text
    print("Searching the internet for similar content...")
    web_results = checker.search_internet(source_text)
    
    # Log the links being checked
    print("\nLinks Checked:")
    for i, (link, _) in enumerate(web_results, 1):
        print(f"Link {i}: {link}")
    
    # Compare source text with web contents
    plagiarism_results = {}
    for i, (link, web_content) in enumerate(web_results):
        similarity = checker.calculate_plagiarism_percentage(checker.preprocess_text(source_text), checker.preprocess_text(web_content))
        plagiarism_results[link] = {
            'similarity': similarity,
            'is_plagiarized': similarity >= checker.threshold
        }
    
    # Print results
    print("\nPlagiarism Check Results:")
    for link, result in plagiarism_results.items():
        print(f"{link}:")
        print(f"Similarity: {result['similarity']:.2%}")
        print(f"Plagiarized: {result['is_plagiarized']}\n")

if __name__ == "__main__":
    main()
