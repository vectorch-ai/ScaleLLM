import openai

client = openai.Client(
    base_url="http://localhost:8080/v1",
    api_key="EMPTY",
)

# List available models
models = client.models.list()
print("==== Available models ====")
for model in models.data:
    print(model.id)

# choose the first model
model = models.data[0].id

stream = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ],
    max_tokens=256,
    stream=True,
)

print(f"==== Model: {model} ====")
for chunk in stream:
    choice = chunk.choices[0]
    delta = choice.delta
    if delta.content:
        print(delta.content, end="")
print()
