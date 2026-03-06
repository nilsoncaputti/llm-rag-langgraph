from fastapi import FastAPI
from app.api.chat import router as chat_router
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="AI Engineering Knowledge Assistant")
app.include_router(chat_router)

@app.get("/")
def root():
    return {"message": "AI Engineering Assistant is running"}