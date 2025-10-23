#For testing purposes only, to be used to check logic before deployment.
from fastapi.testclient import TestClient
from app.main import app

class FakeClient:
    async def chat(self, messages, max_tokens=256, temperature=0.7):
        return {"reply": "Hello from mock!", "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

def test_ai_chat_mock(monkeypatch):
    import app.ai_provider as ap
    monkeypatch.setattr(ap, "AzureOpenAIClient", lambda: FakeClient())
    client = TestClient(app)
    payload = {"messages":[{"role":"user","content":"Say hi!"}]}
    r = client.post("/ai/chat", json=payload)
    assert r.status_code == 200
    assert r.json()["reply"] == "Hello from mock!"
