from scalellm import AsyncLLMEngine, SamplingParams


def main():
    # Create an LLM engine.
    engine = AsyncLLMEngine(model="google/gemma-2b")
    # start the engine loop
    engine.start()

    prompt = input("\n[Prompt]: ")
    while True:
        if prompt == "exit":
            break
        sampling_params = SamplingParams(
            temperature=0, top_p=1.0, max_tokens=100, echo=True
        )
        try:
            output_stream = engine.schedule(
                prompt=prompt,
                sampling_params=sampling_params,
                stream=True,
            )
            for output in output_stream:
                if len(output.outputs) > 0:
                    print(output.outputs[0].text, end="", flush=True)
            print()
        except KeyboardInterrupt:
            # cancel the request
            output_stream.cancel()
            break

        # Get the next prompt.
        prompt = input("\n[Prompt]: ")

    # stop the engine
    engine.stop()


if __name__ == "__main__":
    main()
