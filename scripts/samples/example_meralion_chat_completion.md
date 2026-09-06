curl -X POST https://api.meralion.ai/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MERaLiON/MERaLiON-3-10B",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What is the speaker's emotion?"},
        {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,<BASE64_AUDIO>"}}
      ]
    }],
    "temperature": 0.1
  }

# ChatCompletionResponse Schemas
ChatCompletionResponseCollapse allobject
idstring
objectCollapse allstring
Default"chat.completion"
createdinteger
modelstring
choicesCollapse allarray<object>
ItemsCollapse allobject
indexinteger
messageCollapse allobject
rolestring
contentCollapse all(string | array<object>)
Any ofCollapse all(string | array<object>)
#0string
#1Collapse allarray<object>
Itemsobject
logprobsCollapse all(object | null)
Any ofCollapse all(object | null)
#0object
#1null
finish_reasonstring
usageCollapse allobject
prompt_tokensinteger
completion_tokensinteger
total_tokensinteger