# DocuMind — Document Q&A with Gemini

DocuMind is a lightweight Flask app that summarizes your documents and answers questions using passages retrieved from their contents. Upload one or more files, select a document or search across your library, and inspect the excerpts behind each answer.

## Features

- Upload multiple **PDF, DOCX, TXT, and Markdown** files (20 MB per file).
- Generate a concise document summary with Gemini Flash.
- Retrieve relevant passages using Gemini embeddings and cosine similarity.
- Ask questions about one selected document or the full library.
- See source excerpts and page references with answers.
- Delete documents; their saved passages and embeddings are removed from the local database.
- Keep the Gemini API key on the Flask server, outside browser code.

## How it works

1. Flask extracts text from each uploaded document.
2. The app splits the text into overlapping passages and creates Gemini embeddings.
3. Passages and embeddings are stored in a local SQLite database.
4. For each question, the app embeds the query, selects the five closest passages, and asks Gemini Flash to answer from those passages.

The original uploaded file is removed after processing. Extracted document text, summaries, and embeddings remain locally in `data/rag.sqlite3` until the document is deleted.

## Requirements

- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

## Run locally

Clone the repository and enter the project directory:

```powershell
git clone https://github.com/YOUR-NAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Install the project dependencies with `uv`:

```powershell
uv sync
```

Create your local environment file and add your Gemini key:

```powershell
Copy-Item .env.example .env
notepad .env
```

Set the key in `.env`:

```dotenv
GEMINI_API_KEY=your_gemini_api_key_here
```

Start the Flask server:

```powershell
uv run python app.py
```

Open **http://127.0.0.1:5000** in your browser.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | — | Required Gemini API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model used for summaries and answers |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | Model used for document and question embeddings |
| `DATA_DIR` | `./data` | Location for the local SQLite database |
| `FLASK_DEBUG` | `false` | Enable Flask debug mode for local development only |

Gemini requests use your Google AI Studio account quota and billing settings.

## Project structure

```text
.
├── app.py                 # Flask routes, extraction, embedding, and retrieval
├── templates/
│   └── index.html         # Main page
├── static/
│   ├── app.js             # Upload, document actions, and Q&A interactions
│   └── style.css          # Responsive interface styles
├── .env.example           # Environment variable template (no real key)
├── pyproject.toml         # Project metadata and dependencies
└── uv.lock                # Locked dependency versions
```

## Keep local files private

Do not commit `.env`, `.venv/`, `__pycache__/`, or `data/`. The `.gitignore` excludes them. Never publish your Gemini API key. The `data/` folder contains extracted document contents, summaries, and embeddings; deleting a document in the app removes its indexed content from SQLite.

## Troubleshooting

- **Missing API key:** Make sure `.env` exists beside `app.py` and contains `GEMINI_API_KEY=...`; restart the server after changing it.
- **Upload has no readable text:** Scanned or image-only PDFs may need OCR before upload.
- **Gemini request failed:** Check the API key, network connection, enabled API access, and Google AI Studio quota.
- **Port 5000 is busy:** Stop the other process using it, then restart the app.

## License

No license has been added yet. Add a `LICENSE` file before redistributing this project.
