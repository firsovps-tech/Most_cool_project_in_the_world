import os
import json

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def get_required_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(f"Не задана переменная окружения: {name}")

    return value


def main():
    base_url = get_required_env("OPENAI_BASE_URL")
    api_key = get_required_env("OPENAI_API_KEY")
    model = get_required_env("OPENAI_MODEL")
    project = os.getenv("OPENAI_PROJECT")

    client_kwargs = {
        "base_url": base_url,
        "api_key": api_key,
        "timeout": 30.0,
    }

    if project:
        client_kwargs["project"] = project

    client = OpenAI(**client_kwargs)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Ты отвечаешь только валидным JSON без markdown.",
            },
            {
                "role": "user",
                "content": "Верни JSON с одним полем ok=true.",
            },
        ],
        temperature=0,
    )

    content = response.choices[0].message.content

    print("RAW ANSWER:")
    print(content)

    content = content.strip()

    if content.startswith("```"):
        content = content.strip("`").strip()

        if content.startswith("json"):
            content = content[4:].strip()

    parsed = json.loads(content)

    print()
    print("PARSED:")
    print(parsed)


if __name__ == "__main__":
    main()