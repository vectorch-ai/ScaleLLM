.. _quick_start:

Quick Start
===========

Installation
------------

Install with pip
~~~~~~~~~~~~~~~~
ScaleLLM is available as a Python Wheel package on `PyPI <https://pypi.org/project/scalellm/>`_. You can install it using pip:

.. code-block:: bash

    # Install ScaleLLM with CUDA 12.6 and Pytorch 2.7.0
    $ pip install scalellm

Install other versions
~~~~~~~~~~~~~~~~~~~~~~
If you want to install ScaleLLM with different versions of CUDA and PyTorch, you can use pip by providing the index URL of the desired version.

.. tabs::

    .. tab:: CUDA 12.8

        .. tabs::

            .. tab:: PyTorch 2.7.0

                .. code-block:: bash

                    $ pip install -U scalellm -i https://whl.vectorch.com/cu128/torch2.7.0/

    .. tab:: CUDA 12.6

        .. tabs::

            .. tab:: PyTorch 2.7.0

                .. code-block:: bash

                    $ pip install -U scalellm -i https://whl.vectorch.com/cu126/torch2.7.0/

    .. tab:: CUDA 11.8

        .. tabs::

            .. tab:: PyTorch 2.7.0

                .. code-block:: bash

                    $ pip install -U scalellm -i https://whl.vectorch.com/cu118/torch2.7.0/


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
