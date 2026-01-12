# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import contextlib
import os

from src.query import mcp as raw_mcp_server
from src.query import query_qdrant  # your RAG function

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

# ------------------------
# NORMAL HTTP ENDPOINT FOR UI
# ------------------------
class AskBody(BaseModel):
    question: str

@app.post("/ask")
def ask(body: AskBody):
    try:
        return {"answer": query_qdrant(body.question)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ------------------------
# SIMPLE CHAT UI (NOT ROOT)
# ------------------------
@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
<!doctype html>
<html>
<head>
  <title>RAG Chat</title>
  <style>
    body { font-family: Arial; max-width: 800px; margin: 40px auto; }
    textarea { width: 100%; height: 80px; }
    button { padding: 8px 16px; margin-top: 8px; }
    pre { background: #111; color: #0f0; padding: 12px; white-space: pre-wrap; }
  </style>
</head>
<body>
  <h2>Chat with RAG</h2>

  <textarea id="q" placeholder="Ask a question..."></textarea>
  <button onclick="ask()">Ask</button>

  <h3>Answer</h3>
  <pre id="a"></pre>

  <script>
    async function ask() {
      const question = document.getElementById("q").value;
      const res = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question })
      });
      const data = await res.json();
      document.getElementById("a").textContent = data.answer || data.error;
    }
  </script>
</body>
</html>
"""

# ------------------------
# MCP OWNS ROOT (AS YOU REQUIRE)
# ------------------------
app.mount("/", raw_mcp_server.streamable_http_app())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
