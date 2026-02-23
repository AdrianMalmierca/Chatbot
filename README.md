# Chatbot
This project implements a full Retrieval-Augmented Generation (RAG) system that allows users to query academic publication data extracted from the scientific portal of the Universidad Pontificia de Salamanca.

The source data was obtained from:

https://portalcientifico.upsa.es/

The website was crawled using Apache Nutch, generating a dump that was later transformed and cleaned into a structured JSON format suitable for semantic indexing and retrieval.

## Before start
The files faiss_index.bin, doc_metadata.pkl, embeddings.npy and doc_metadata.pkl are generated when the Main.py and Embeddings.py code is executed. These are very large files, so they are not uploaded to the repository, but they are vital for the chatbot to function.

### Theoretical background: RAG (Retrieval-Augmented Generation)
Retrieval-Augmented Generation (RAG) is an advanced NLP architecture that combines retrieval-based methods with generative language models. It was introduced to address the limitation of pure language models: they can generate fluent text but often hallucinate or produce inaccurate information if the knowledge is not embedded in their parameters.

The RAG paradigm separates the task into two complementary stages:
1. Retrieval Stage
    - Given a user query, the system retrieves the most relevant documents from a large corpus.
    - This is usually done via vector similarity search:
        - Documents are encoded into high-dimensional embeddings (numerical representations of semantic content).
        - The query is encoded using the same embedding model.
        - Similarity metrics (e.g., cosine similarity or L2 distance) identify documents closest in semantic space to the query.

    - Advantages:
        - Keeps knowledge up-to-date without retraining the language model.
        - Reduces hallucinations by grounding answers in real data.

2. Generation Stage
    - The retrieved documents are provided as context to a large language model (LLM), which generates a natural-language answer.
    - The model is instructed to rely strictly on the provided context.
    - Prompts often include rules like:
        - “Do not invent information.”
        - “Integrate document content naturally into paragraphs.”
        - “List authors or collaborations only if present in the documents.”

### Methodological justification
1. Why RAG for Academic Assistance?
    - Dynamic Knowledge Base
    - Academic research grows rapidly. Pre-trained LLMs alone may be outdated.
    - Using a retrieval system, the assistant can access the most recent publications without retraining the model.

2. Handling Large Document Sets
    - Scientific portals contain thousands of publications.
    - Embeddings + FAISS allow efficient similarity search across large corpora, ensuring fast retrieval.

3. Precision over Fluency
    - Classical LLMs excel in fluency but may hallucinate factual details.
    - By grounding responses in retrieved articles, RAG improves factual correctness, which is crucial for academic use.

4. Scalability and Modularity
    - Retrieval and generation stages are decoupled.
    - Corpus can be updated independently of the LLM.
    - The embedding model and similarity metric can be swapped for improved performance.

5. Token Management & Multi-turn Conversations
    - Academic queries can be lengthy.
    - Limiting context tokens ensures that relevant information fits the LLM input window.
    - Conversation history can be truncated intelligently to retain continuity without exceeding model limits.

6. Explainability
    - The system can reference which documents were used to generate each response.
    - This increases trustworthiness and allows verification of sources.

## Structure
Angular Frontend
        ↓
FastAPI Backend
        ↓
RAG Engine (FAISS + OpenAI Embeddings + GPT-4)

## Main components
- Frontend: Angular (chat-based UI)
- Backend: FastAPI (REST API)
- Vector Store: FAISS
- Embeddings Model: text-embedding-3-small
- LLM: gpt-4-turbo
- Data Source: Scientific portal (crawled and transformed into JSON)
```
Chatbot/
│
├── frontend/
│       ├── src/
│           ├── app/
│               ├── chat.component.ts
│               ├── chat.component.html
│               ├── chat.component.css
│               ├── services/
│                       └── chat.service.ts
│
|──es/
|   ├──upsa/
|        ├──tfg/
│            ├── main.py
│            ├── ai.py
│            ├── embeddings.py
│            ├── faiss_index.bin
│            ├── doc_metadata.pkl
│            ├── embeddings.npy
|            └──output_clean3.json
````

## Data pipeline
1. Data Extraction
2. The scientific portal was crawled using Apache Nutch.
3. The generated dump was cleaned and transformed into a structured JSON file.
4. Each document contains:
    - Title
    - Authors
    - Abstract / Summary
    - Year of publication
    - ISBN/ISSN
    - Conference
    - Type of publication

Example structure:
{
  "title": "...",
  "authors": ["Author1", "Author2"],
  "summary": "...",
  "year_of_publication": "...",
  "isbn_issn": "...",
  "congress": "...",
  "type_of_publication": "..."
}

## Embedding generation
Script: embeddings.py

### Process:
1. Load cleaned JSON documents.
2. Build a textual representation combining:
    - Title
    - Summary
    - Authors
3. Generate embeddings using:
    text-embedding-3-small

4. Store vectors in a FAISS index (IndexFlatL2).
5. Persist:
    - faiss_index.bin
    - doc_metadata.pkl
    - embeddings.npy

Run:
```bash
python embeddings.py
```

## RAG engine
Core file: Ai.py

### Retrieval phase
1. User submits a query.
2. The query is embedded.
3. FAISS performs similarity search.
4. Most relevant documents are selected.

### Context construction
The system builds a structured context including:
- Title
- Authors
- Year
- Conference
- Publication type
- Summary
The context is token-limited to prevent overflow.

### Generation phase
The context is sent to gpt-4-turbo with strict instructions:
- Answer only using provided documents.
- Do not hallucinate information.
- Write in natural narrative language.
- Avoid bullet points and metadata formatting.
- Explicitly state when information is unavailable.

## Token management
To avoid exceeding model limits:
- Maximum context: 18,000 tokens
- Conversation history is dynamically trimmed
- System instructions are preserved
- Old messages are removed when necessary
This ensures stable multi-turn conversations.

## Backend – FastAPI
File: Main.py
Exposes endpoint:
POST /chat

Example request:
{
  "text": "Which authors work in artificial intelligence?"
}

Example response:
{
  "response": "..."
}

CORS is enabled for Angular integration.

Run backend:
```bash
uvicorn main:app --reload --port 8000
```

## Frontend – Angular

### ChatComponent
Features:
- Real-time chat interface
- Message history
- Loading indicator
- Error handling
- Unique session ID per session

### ChatService
Sends requests to:
http://localhost:8000/chat

Payload:
{
  "question": "...",
  "session_id": "uuid"
}

Run frontend:
```bash
ng serve
```

Access:
http://localhost:4200

## System capabilities
The assistant can:
- Retrieve publications by topic
- List all publications of a specific author
- Identify co-authorship relationships
- Count researchers working in a given area
- Confirm collaborations between two authors
- Explicitly state when data is insufficient
The model is explicitly instructed to rely exclusively on retrieved documents.

## Running the full system

1. Clone the repository
git clone https://github.com/AdrianMalmierca/Chatbot

### Backend
2. Access to the root
```bash
cd Chatbot
```
3. Run once to create the index
```bash
python embeddings.py        
```
4. Starts a local server using Uvicorn, an ASGI web server for Python.

Loads the app object from the main.py file, which contains the FastAPI application instance.

Runs the server on http://127.0.0.1:8000 by default.

Enables auto-reload (--reload) so that the server automatically restarts whenever you modify your code.
```bash
uvicorn main:app --reload
```

### Frontend
2. Access to the frontend
```bash
cd frontend
```

3. Run the Angular code
```bash
ng serve
```

Open:
http://localhost:4200

## Future improvements
1. Persistent user sessions
2. Semantic re-ranking
3. Filtering by year or publication type
4. Authentication system
5. Dockerized deployment
6. Replace FAISS with managed vector DB (Pinecone, Weaviate, etc.)
7. API key management via secure vault
8. Improved author disambiguation

## Technology stack
1. Angular
2. TypeScript
3. FastAPI
4. Python
5. FAISS
6. OpenAI API
7. NumPy
8. Pickle
9. Apache Nutch

# What did I learn?
This project has been a completely challenge cause it's the first time I have done something with AI. Previously I have learned Python but mmore basic knowledge, like some operations or charts. But with this project first of all I have learned how to obtain all the information from a web page, cause before I heard about crawler but I had never done something related. I also have learned how tu use APIs for this type of projects and also what's the logic that the chatbot follows. I have learned to create better prompts so the chatbot could answer better and also why are important the embeddings and how they work. Finally I have learned to connect backend and frontend in different codes cause has been the first time I have done something like that. Although this project has taken a lot of time, Im so grateful with the result.

## Author
Adrián Martín Malmierca

Computer Engineer & Mobile Applications Master's Student