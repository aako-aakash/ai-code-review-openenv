from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "AI Code Review Server Running 🚀"}


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)