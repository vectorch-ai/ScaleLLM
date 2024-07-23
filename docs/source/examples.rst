.. _examples:

Examples
========

You can use ScaleLLM for offline batch completions or online inference. Below are some examples to help you get started. More examples can be found in the `examples <https://github.com/vectorch-ai/ScaleLLM/tree/main/examples>`_ folder.

Chat Completion
---------------

Start the REST API server with the following command:

.. code-block:: bash

   $ python3 -m scalellm.serve.api_server --model=meta-llama/Meta-Llama-3.1-8B-Instruct

You can query the chat completions using `curl` or the OpenAI client:

.. tabs::

   .. group-tab:: curl

      .. code-block:: bash

         $ curl http://localhost:8080/v1/chat/completions \
           -H "Content-Type: application/json" \
           -d '{
             "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
             "messages": [
               {
                 "role": "system",
                 "content": "You are a helpful assistant."
               },
               {
                 "role": "user",
                 "content": "Hello!"
               }
             ]
           }'

   .. group-tab:: openai

      .. code-block:: python

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

         # Choose the first model
         model = models.data[0].id

         stream = client.chat.completions.create(
             model=model,
             messages=[
                 {"role": "system", "content": "You are a helpful assistant."},
                 {"role": "user", "content": "Hello"},
             ],
             stream=True,
         )

         print(f"==== Model: {model} ====")
         for chunk in stream:
             choice = chunk.choices[0]
             delta = choice.delta
             if delta.content:
                 print(delta.content, end="")
         print()

Completions
-----------

Start the REST API server with the following command:

.. code-block:: bash
  
   $ python3 -m scalellm.serve.api_server --model=meta-llama/Meta-Llama-3.1-8B

You can query the completions using `curl` or the OpenAI client:

.. tabs::

   .. group-tab:: curl

      .. code-block:: bash

         $ curl http://localhost:8080/v1/completions \
          -H "Content-Type: application/json" \
          -d '{
            "model": "meta-llama/Meta-Llama-3-8B",
            "prompt": "hello",
            "max_tokens": 32,
            "temperature": 0.7,
            "stream": true
          }'

   .. group-tab:: openai

      .. code-block:: python

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

         # Choose the first model
         model = models.data[0].id

         stream = client.completions.create(
             model=model,
             prompt="hello",
             max_tokens=32,
             temperature=0.7,
             stream=True,
         )

         print(f"==== Model: {model} ====")
         for chunk in stream:
             choice = chunk.choices[0]
             if choice.text:
                 print(choice.text, end="")
         print()
