from scalellm import AsyncLLMEngine, Message, SamplingParams


def main():
    # Create an LLM engine.
    with AsyncLLMEngine(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct", devices="cuda"
    ) as engine:
        sampling_params = SamplingParams(temperature=0.7, max_tokens=1000)

        messages = []
        system_prompt = input("\n[System]: ")
        # append the system message
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))

        while True:
            # Get the next prompt.
            prompt = input("\n[User]: ")
            if not prompt:
                continue
            if prompt == "exit" or prompt == "quit":
                break

            # append the user message
            messages.append(Message(role="user", content=prompt))

            try:
                output_stream = engine.schedule_chat(
                    messages=messages,
                    sampling_params=sampling_params,
                    stream=True,
                )
                assistant_response = ""
                print("\n[Assistant]: ", end="", flush=True)
                for output in output_stream:
                    if len(output.outputs) > 0:
                        response = output.outputs[0].text
                        assistant_response += response
                        print(response, end="", flush=True)
                print()
            except KeyboardInterrupt:
                # cancel the request
                output_stream.cancel()
                break

            # append the assistant message
            messages.append(Message(role="assistant", content=assistant_response))


if __name__ == "__main__":
    main()
