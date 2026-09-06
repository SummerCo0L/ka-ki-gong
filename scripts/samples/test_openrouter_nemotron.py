from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="<OPENROUTER_API_KEY>",
)

embedding = client.embeddings.create(
  extra_headers={
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-OpenRouter-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  model="nvidia/nemotron-3-embed-1b:free",
  input="Your text string goes here",
  # input: ["text1", "text2", "text3"] # batch embeddings also supported!
  encoding_format="float"
)
print(embedding.data[0].embedding)