# Gemini PDF Question Answering System

### This Retrieval-Augmented Generation (RAG) bot allows users to ask questions related to a PDF file through a command-line interface. By using advanced language models from Google GenAI and text processing with Langchain, the bot answers queries based on the content of the PDF.

## Pre-Requisites
* Python 3.10 or higher
* `pdfplumber` for PDF text extraction
* `Langchain` for RAG-based processing
* `Google Generative AI` for question answering

## Installation
1. Clone this repository to your local machine.
   ```bash
   git clone [<repository_url>](https://github.com/Darshbir/Command-Line-RAG-Bot)
   cd Command-Line-RAG-Bot
   ```

2. Create a virtual environment and activate it.
   - For Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - For macOS/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. Install the required packages by running:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory of the project and add your Google API key as shown below or refer `.env.example`:

   ```python
   GOOGLE_API_KEY=<your_google_api_key>
   ```
5. Add the file you want to address the query with in the local  folder.
5. Run the application by executing:
   ```bash
   python bot.py <pdf_file_name.pdf> "Your question here"
   ```

## Usage
1. Just copy the path of the file you want to query.
2. Submit your query in the following format in the command line:
   ```bash
   python bot.py <pdf_file_name.pdf> "Your question here"
   ```

3. The application will use Langchain and Google Generative AI to find the answer in the context of the PDF and display it.

## Code Structure
* `bot.py`: The main application that handles PDF extraction, text splitting, and querying using Google GenAI.
* `get_pdf_text`: A function to extract text from the uploaded PDF files using `pdfplumber`.
* `get_text_chunks`: A function to split the extracted text into smaller chunks for efficient processing.
* `get_vector_store`: A function to create a vector database from the text chunks for question-answering.
* `main`: The main function that sets up the application, processes the PDF, and retrieves the answer to the query.

## Example

### 1. Input Command:
```bash
python bot.py sample_document.pdf "What is the attention mechanism in neural networks?"
```

### 2. Output:
```text
Answer:
The provided text states that the underlying idea of transformer models is to represent the input sequence through a set of encoders...

Citations:
- Page 2: Transformer models use encoders with attention mechanisms (Vaswani et al., 2017).
- Page 21: Related work in machine learning...
```

## Troubleshooting

- **Missing Dependencies**:  
  If you encounter issues with missing dependencies, ensure that you've activated the virtual environment and run:
  ```bash
  pip install -r requirements.txt
  ```

- **Invalid API Key**:  
  Ensure your Google API key is valid and properly set in the `.env` file.

- **PDF Extraction Issues**:  
  If the text extraction isn't working, the PDF might be encrypted or in a non-standard format. Ensure the file is readable by `pdfplumber`.

## Future Enhancements
- Removing the excessive warnings appearing at the beginning and preventing downloading the stopwords everytime the program is run
- Improve text extraction accuracy.
- Add multi-step reasoning for complex question answering.
- Implement a web-based interface for easier interaction.

