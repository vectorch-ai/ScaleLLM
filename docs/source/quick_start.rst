.. _quick_start:

Quick Start
===========

Installation
------------

Install with pip
~~~~~~~~~~~~~~~~
ScaleLLM is available as a Python Wheel package on `PyPI <https://pypi.org/project/scalellm/>`_. You can install it using pip:

.. code-block:: bash

    # Install ScaleLLM with CUDA 12.4 and Pytorch 2.5.1
    $ pip install scalellm

Install other versions
~~~~~~~~~~~~~~~~~~~~~~
If you want to install ScaleLLM with different versions of CUDA and PyTorch, you can use pip by providing the index URL of the desired version.

.. tabs::

    .. tab:: CUDA 12.4

        .. tabs::

            .. tab:: PyTorch 2.5.1

                .. code-block:: bash

                    $ pip install -U scalellm -i https://whl.vectorch.com/cu124/torch2.5.1/

            .. tab:: PyTorch 2.4.1

                .. code-block:: bash

                    $ pip install -U scalellm -i https://whl.vectorch.com/cu124/torch2.4.1/

    .. tab:: CUDA 12.1

        .. tabs::

            .. tab:: PyTorch 2.5.1

                .. code-block:: bash

                    $ pip install -U scalellm -i https://whl.vectorch.com/cu121/torch2.5.1/

            .. tab:: PyTorch 2.4.1

                .. code-block:: bash

                    $ pip install -U scalellm -i https://whl.vectorch.com/cu121/torch2.4.1/

    .. tab:: CUDA 11.8

        .. tabs::

            .. tab:: PyTorch 2.5.1

                .. code-block:: bash

                    $ pip install -U scalellm -i https://whl.vectorch.com/cu118/torch2.5.1/

            .. tab:: PyTorch 2.4.1

                .. code-block:: bash

                    $ pip install -U scalellm -i https://whl.vectorch.com/cu118/torch2.4.1/


Build from source
~~~~~~~~~~~~~~~~~
If no wheel package is available for your configuration, you can build ScaleLLM from source code. Clone the repository and install it locally using the following commands:

.. code-block:: bash

    $ git clone --recursive https://github.com/vectorch-ai/ScaleLLM.git
    $ cd ScaleLLM
    $ python3 setup.py bdist_wheel
    $ pip install dist/scalellm-*.whl


Inference
-------

You can use ScaleLLM for offline batch inference or online distributed inference.

OpenAI-Compatible Server
~~~~~~~~~~~~~~~~~~~~~~~~
To start a server that is compatible with the OpenAI API, run the following command:

.. code-block:: bash

    $ python3 -m scalellm.serve.api_server --model=meta-llama/Meta-Llama-3.1-8B-Instruct
