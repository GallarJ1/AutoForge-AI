import os, httpx

# --- Read Azure OpenAI environment variables ---
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")


class AzureOpenAIClient:
    def __init__(self):
        # Ensure all required variables exist before continuing
        if not (AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY and AZURE_OPENAI_DEPLOYMENT):
            raise RuntimeError("Missing Azure OpenAI env vars")

        # Normalize and store for later
        self.base = AZURE_OPENAI_ENDPOINT.rstrip("/")
        self.key = AZURE_OPENAI_KEY
        self.api_version = AZURE_OPENAI_API_VERSION
        self.deployment = AZURE_OPENAI_DEPLOYMENT

    async def chat(self, messages: list[dict], max_tokens: int = 256, temperature: float = 0.7):
        """
        Sends a chat completion request to your Azure OpenAI deployment.
        This wraps the REST API call so your FastAPI endpoint can simply call client.chat()
        """
        url = f"{self.base}/openai/deployments/{self.deployment}/chat/completions"
        params = {"api-version": self.api_version}
        headers = {"api-key": self.key, "Content-Type": "application/json"}
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "n": 1}

        timeout = httpx.Timeout(10.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, headers=headers, params=params, json=body)
            r.raise_for_status()
            data = r.json()

            choice = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return {"reply": choice, "usage": usage}
