# main.py
import contextlib
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from src.query import mcp as raw_mcp_server
from src.query import query_qdrant  # <-- your tool function

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(raw_mcp_server.session_manager.run())
        yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down later if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# 1) User-facing REST API
# -----------------------
class AskBody(BaseModel):
    question: str

@app.post("/ask")
def ask(body: AskBody):
    try:
        answer = query_qdrant(body.question)  # calls your RAG tool directly
        return {"answer": answer}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# -----------------------
# 2) Simple frontend page
# -----------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <title>RAG Chat</title>
    <style>
      body { font-family: Arial, sans-serif; max-width: 900px; margin: 30px auto; }
      textarea { width: 100%; height: 80px; padding: 10px; }
      button { padding: 10px 16px; margin-top: 10px; cursor: pointer; }
      pre { background: #0b0b0b; color: #00ff7f; padding: 16px; white-space: pre-wrap; border-radius: 10px; }
      .row { display:flex; gap:10px; align-items:center; }
      .muted { color:#666; font-size: 12px; }
    </style>
  </head>
  <body>
    <h2>RAG Q&A</h2>
    <textarea id="q" placeholder="Ask a question..."></textarea>
    <div class="row">
      <button id="btn">Ask</button>
      <span id="status" class="muted"></span>
    </div>
    <h3>Answer</h3>
    <pre id="a"></pre>

    <script>
      const btn = document.getElementById("btn");
      const q = document.getElementById("q");
      const a = document.getElementById("a");
      const status = document.getElementById("status");

      btn.onclick = async () => {
        a.textContent = "";
        status.textContent = "Thinking...";
        btn.disabled = true;

        try {
          const res = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: q.value })
          });
          const data = await res.json();
          if (!res.ok) throw new Error(data.error || "Request failed");
          a.textContent = data.answer;
          status.textContent = "";
        } catch (e) {
          status.textContent = "Error";
          a.textContent = String(e);
        } finally {
          btn.disabled = false;
        }
      };
    </script>
  </body>
</html>
"""

# -----------------------
# 3) MCP endpoint for tools (Inspector/Cursor)
# -----------------------
# IMPORTANT: mount MCP app at /mcp (not "/") so your / and /ask routes work
app.mount("/mcp", raw_mcp_server.streamable_http_app())

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="debug")
