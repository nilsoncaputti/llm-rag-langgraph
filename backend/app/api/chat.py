from app.agents.streaming_rag_agent import stream_answer
from fastapi import APIRouter
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from app.agents.rag_agent import run_agent

router = APIRouter()

class QuestionRequest(BaseModel):
    question: str


@router.post("/ask")
def ask_question(req: QuestionRequest):

    result = run_agent(req.question)

    return {
        "answer": result["answer"],
        "sources": result["sources"]
    }

@router.post("/stream")
def stream_chat(req: QuestionRequest):

    return StreamingResponse(
        stream_answer(req.question),
        media_type="text/plain"
    )