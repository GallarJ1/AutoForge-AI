from fastapi.testclient import TestClient
import app.main as m  # import the module, not just app

class FakeClient:
    async def chat(self, messages, max_tokens=256, temperature=0.7):
        return {"reply": "Hello from mock!", "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

def test_ai_chat_mock(monkeypatch):
    # Patch where it's *used* (app.main), not where it's defined
    monkeypatch.setattr(m, "AzureOpenAIClient", lambda: FakeClient())
    client = TestClient(m.app)
    payload = {"messages": [{"role": "user", "content": "Say hi!"}]}
    r = client.post("/ai/chat", json=payload)
    assert r.status_code == 200
    assert r.json()["reply"] == "Hello from mock!"
