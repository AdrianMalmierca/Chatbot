from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Habilitar CORS para permitir peticiones desde Angular
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes restringir esto a http://localhost:4200 si lo prefieres
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo de datos para recibir mensajes
class Message(BaseModel):
    text: str

@app.post("/chat")
async def chat_response(message: Message):
    response_text = f"Recibí tu mensaje: {message.text}"  # Aquí iría la lógica del chatbot
    return {"response": response_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
