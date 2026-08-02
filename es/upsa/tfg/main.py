from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from Ai import responder, docs
import os

app = FastAPI()

# Habilitar CORS para permitir peticiones desde Angular
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "https://chatbot-sigma-two-27.vercel.app",
        "https://chatbot-git-main-adrians-projects-9d5c028d.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Data model, we use basemodel to validate
class Message(BaseModel):
    question: str = Field(..., min_length=1, max_length=500) #... = compulsory
    session_id: str = Field(..., min_length=1, max_length=100)

@app.post("/chat")
async def chat_response(message: Message):
    respuesta = responder(message.question, message.session_id)
    return {"response": respuesta}

@app.get("/health")
async def health_check():
    return {"status": "ok", "documentos_cargados": len(docs)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
