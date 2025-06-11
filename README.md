# PDF Document FAQ System

A sophisticated AI-powered question-answering system that allows users to upload PDF documents and ask natural language questions about their content. The system uses advanced retrieval techniques and language models to provide accurate answers with confidence scores.

## 🌟 Features

- **PDF Document Processing**: Load and process PDF documents with automatic text extraction
- **Hybrid Retrieval**: Combines semantic search (FAISS) with keyword-based search (BM25) for optimal results
- **AI-Powered Answers**: Uses Groq's LLaMA 3 model for generating natural language responses
- **Confidence Scoring**: Provides confidence scores for each answer to help assess reliability
- **Multiple Interfaces**: 
  - Interactive Streamlit web application
  - Command-line interface (CLI)
  - Google Colab integration with ngrok tunneling
- **Performance Metrics**: Tracks processing time and confidence levels for each query
- **Real-time Processing**: Fast document embedding and question answering

## 🛠 Technology Stack

- **Language Model**: Groq LLaMA 3 (8B parameters)
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **Vector Database**: FAISS for semantic search
- **Text Processing**: LangChain framework
- **Web Interface**: Streamlit
- **PDF Processing**: PyPDF2 via LangChain
- **Retrieval**: Ensemble of semantic and keyword-based retrievers

## 📋 Prerequisites

- Python 3.7+
- Groq API key (free tier available)
- ngrok account and auth token (for Colab deployment)

## 🚀 Installation

### Method 1: Local Installation

```bash
# Install required packages
pip install streamlit langchain faiss-cpu sentence-transformers PyPDF2 nltk requests pandas numpy

# Clone or download the code
# Set up your API keys (see Configuration section)
```

### Method 2: Google Colab (Recommended for beginners)

1. Open the provided notebook in Google Colab
2. Run all cells in sequence
3. The system will automatically install dependencies

## ⚙️ Configuration

### 1. Groq API Key Setup

Get your free API key from [Groq Console](https://console.groq.com/):

```python
# Replace in the code:
GROQ_API_KEY = 'your_actual_groq_api_key_here'
```

### 2. Ngrok Setup (for Colab)

Get your auth token from [ngrok Dashboard](https://dashboard.ngrok.com/get-started/your-authtoken):

```python
# Replace in the code:
NGROK_AUTHTOKEN = "your_actual_ngrok_token_here"
```

## 🖥 Usage

### Streamlit Web Interface

#### Local Development:
```bash
streamlit run streamlit_app.py
```

#### Google Colab:
Run all cells in the notebook - the system will provide a public URL via ngrok.

**Using the Web Interface:**
1. Enter the PDF file path in the sidebar
2. Click "Load PDF" to process the document
3. Ask questions in the text area
4. View answers with confidence scores and processing times

### Command Line Interface

```python
# Run the CLI version
main_cli()
```

**Using the CLI:**
1. Enter the PDF file path when prompted
2. Ask questions interactively
3. Type 'quit' to exit

## 📊 System Architecture

### Core Components

1. **PDFDocumentFAQSystem**: Main class handling document processing and Q&A
2. **GroqLLM**: Custom LangChain wrapper for Groq API integration
3. **Hybrid Retrieval**: Combines FAISS semantic search with BM25 keyword matching
4. **Confidence Scoring**: Multi-factor algorithm assessing answer reliability

### Processing Pipeline

```
PDF Upload → Text Extraction → Chunking → Embedding Generation → 
Vector Store Creation → Question Processing → Hybrid Retrieval → 
LLM Answer Generation → Confidence Calculation → Response Display
```

### Confidence Scoring Algorithm

The system calculates confidence based on:
- **Document Factor** (30%): Number of relevant source documents found
- **Similarity Factor** (40%): Semantic similarity between question and retrieved content  
- **Length Factor** (30%): Completeness of the generated answer
- **Uncertainty Penalty**: Reduction for phrases indicating uncertainty

## 🎯 Confidence Score Interpretation

- **🟢 0.8-1.0**: High Confidence - Answer likely very accurate
- **🟡 0.6-0.8**: Medium Confidence - Reasonably accurate, may need verification
- **🟠 0.4-0.6**: Low Confidence - Uncertain, verify with original document
- **🔴 0.0-0.4**: Very Low Confidence - Information may not be available

## 📁 File Structure

```
├── streamlit_app.py          # Streamlit web application
├── main_notebook.ipynb       # Google Colab notebook
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🔧 Customization Options

### Model Configuration
```python
# Change the language model
self.llm = GroqLLM(api_key=groq_api_key, model='mixtral-8x7b-32768')

# Adjust chunk size for document processing
self.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,  # Increase for longer contexts
    chunk_overlap=300,
    length_function=len
)
```

### Retrieval Tuning
```python
# Adjust retrieval weights
self.retriever = EnsembleRetriever(
    retrievers=[semantic_retriever, bm25_retriever],
    weights=[0.8, 0.2]  # Favor semantic search more
)
```

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure Groq API key is correctly set
   - Check key validity and rate limits

2. **PDF Loading Failures**
   - Verify file path is correct
   - Ensure PDF is not password-protected
   - Check file permissions

3. **Memory Issues**
   - Reduce chunk_size for large documents
   - Process smaller PDF files
   - Clear session state in Streamlit

4. **Ngrok Connection Issues**
   - Verify ngrok auth token
   - Check firewall settings
   - Restart the tunnel

### Performance Optimization

- **For Large PDFs**: Increase chunk_size and reduce chunk_overlap
- **For Better Accuracy**: Increase number of retrieved documents (k parameter)
- **For Faster Processing**: Use smaller embedding models

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request


## 🔄 Version History

- **v1.0**: Initial release with basic PDF Q&A functionality
- **v1.1**: Added confidence scoring and performance metrics
- **v1.2**: Implemented hybrid retrieval system
- **v1.3**: Added Streamlit web interface and ngrok integration

## 🙏 Acknowledgments

- Groq for providing fast LLM inference
- LangChain for the RAG framework
- Streamlit for the web interface
- SentenceTransformers for embeddings
- FAISS for efficient vector search
