import requests

from config import OLLAMA_API_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_SECONDS


def main() -> None:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": "Explain crude oil price risk for an oil company in 3 sentences.",
        "stream": False,
    }

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json=payload,
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
    except requests.RequestException as error:
        print(f"Ollama request failed: {error}")
        return

    if response.status_code == 200:
        result = response.json()
        print("\nOllama response:\n")
        print(result["response"])
    else:
        print("Error:", response.status_code)
        print(response.text)


if __name__ == "__main__":
    main()