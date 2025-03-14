import asyncio

from scalellm import AsyncLLMEngine, SamplingParams


async def main():
    # Create an LLM engine.
    with AsyncLLMEngine(model="meta-llama/Meta-Llama-3.1-8B", devices="cuda") as engine:
        sampling_params = SamplingParams(
            temperature=0, top_p=1.0, max_tokens=100, echo=True
        )

        while True:
            # Get the next prompt.
            prompt = input("\n[Prompt]: ")
            if not prompt:
                continue
            if prompt == "exit":
                break
            try:
                output_stream = await engine.schedule_async(
                    prompt=prompt,
                    sampling_params=sampling_params,
                    stream=True,
                )
                async for output in output_stream:
                    if len(output.outputs) > 0:
                        print(output.outputs[0].text, end="", flush=True)
                print()
            except KeyboardInterrupt:
                # cancel the request
                output_stream.cancel()
                break


if __name__ == "__main__":
    asyncio.run(main())
