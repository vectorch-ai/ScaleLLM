from scalellm import AsyncLLMEngine, SamplingParams


def main():
    # Create an LLM engine.
    engine = AsyncLLMEngine(model="google/gemma-2b")
    # start the engine loop
    engine.start()

    prompt = input("Enter a prompt: ")
    while True:
        if prompt == "exit":
            break
        sampling_params = SamplingParams(
            temperature=0, top_p=1.0, max_tokens=100, echo=True
        )
        output_stream = engine.schedule(
            prompt=prompt,
            sampling_params=sampling_params,
            stream=True,
        )
        for output in output_stream:
            if len(output.outputs) > 0:
                print(output.outputs[0].text, end="", flush=True)
        print()

        # Get the next prompt.
        prompt = input("Enter a prompt: ")

    # stop the engine
    engine.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
