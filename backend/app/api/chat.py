from fastapi import APIRouter
from pydantic import BaseModel
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