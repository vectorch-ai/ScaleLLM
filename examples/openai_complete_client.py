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

stream = client.completions.create(
    model=model,
    prompt="hello",
    max_tokens=32,
    stream=True,
)

print(f"==== Model: {model} ====")
for chunk in stream:
    choice = chunk.choices[0]
    if choice.text:
        print(choice.text, end="")
print()
