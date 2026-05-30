import asyncio
import httpx

async def test():
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            "http://127.0.0.1:8000/api/generate-campaign",
            json={
                "user_intent": "Analyze the README for the repository InsightsDSA and create a deep-dive technical launch campaign highlighting its architecture and value proposition.",
                "tone": "Professional"
            }
        )
        print(resp.status_code)
        try:
            print(resp.json())
        except:
            print(resp.text)

if __name__ == "__main__":
    asyncio.run(test())
