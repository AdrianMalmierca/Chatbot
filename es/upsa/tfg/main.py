from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from Ai import responder

app = FastAPI()

# Habilitar CORS para permitir peticiones desde Angular
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo de datos para recibir mensajes
class Message(BaseModel):
    question: str
    session_id: str

@app.post("/chat")
async def chat_response(message: Message):
    respuesta = responder(message.question, message.session_id)
    return {"response": respuesta}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
