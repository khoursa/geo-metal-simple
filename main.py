from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Geo Metal Detector API is working"}
