from fastapi import FastAPI
import uvicorn
from router import upload_router
from router import retrieve_router

app = FastAPI(title="PDF Upload and Process API")

app.include_router(upload_router.router, prefix="/files")
app.include_router(retrieve_router.router, prefix="/retrieve")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=4545, reload=True)