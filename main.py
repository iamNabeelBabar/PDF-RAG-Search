from fastapi import FastAPI
import uvicorn
from router.upload_router import router as upload_router
from router.retrieve import router as retrieve_router
import os

app = FastAPI(title="PDF Upload and Process API")

app.include_router(upload_router, prefix="/files")
app.include_router(retrieve_router, prefix="/retrieve")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 4545))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
