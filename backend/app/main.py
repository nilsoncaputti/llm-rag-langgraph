from fastapi import FastAPI

app = FastAPI(title="AI Engineering Knowledge Assistant")

@app.get("/")
def root():
    return {"message": "AI Engineering Assistant is running"}