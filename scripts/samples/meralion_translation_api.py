import base64
import os
import pathlib
import requests
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

audio_b64 = base64.b64encode(
    (REPO_ROOT / "artefacts" / "2_hokkien.wav").read_bytes()
).decode("ascii")

response = requests.post(
    "https://api.meralion.ai/v1/audio/translations",
    headers={
        "Authorization": f"Bearer {os.environ['MERALION_API_KEY']}",
        "Content-Type": "application/json",
    },
    json={
        "audio_url": f"data:audio/wav;base64,{audio_b64}",
        "model": "MERaLiON/MERaLiON-3-3B-ASR-CTM",
        "stream": False,
        "translation_params": {
            "source_language": "",
            "target_language": "English",
        },
    },
    timeout=90,
)

print(response.status_code)
print(response.text)
response.raise_for_status()

message = response.json()["choices"][0]["message"]
print(message["content"])