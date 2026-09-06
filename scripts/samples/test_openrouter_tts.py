from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="<OPENROUTER_API_KEY>",
)

with client.audio.speech.with_streaming_response.create(
  extra_headers={
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-OpenRouter-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  model="qwen/qwen-audio-3.0-tts-flash",
  input="Hello! This is a text-to-speech test.",
  voice="loongjohn"
) as response:
  response.stream_to_file("output.mp3")