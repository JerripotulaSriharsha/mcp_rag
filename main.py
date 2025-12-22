from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import contextlib
from src.query import mcp as raw_mcp_server

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(raw_mcp_server.session_manager.run())
        yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/rag", raw_mcp_server.streamable_http_app())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=10000, log_level="debug")