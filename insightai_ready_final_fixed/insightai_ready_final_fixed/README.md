# InsightAI

InsightAI is a semantic search engine that allows users to upload PDF documents and ask natural language questions about their content. It leverages OpenAI's embeddings and FAISS for efficient document search.

## Features

- Upload one or multiple PDFs
- Ask natural language questions about their content
- Backend: FastAPI + OpenAI + FAISS
- Frontend: Simple HTML/JS UI

## Requirements

- Python 3.9+
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/insightai.git
cd insightai
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Add your OpenAI API key:
Create a `.env` file in the root directory with the following content:
```
OPENAI_API_KEY=your_openai_api_key
```

## Run the App

```bash
uvicorn app.main:app --reload
```

Visit `http://127.0.0.1:8000` to access the application.

## Docker

To run with Docker:
```bash
docker build -t insightai .
docker run -p 8000:8000 insightai
```

## Example PDFs

Add sample PDFs to the `test_pdfs/` folder to try the application.

## License

MIT License
