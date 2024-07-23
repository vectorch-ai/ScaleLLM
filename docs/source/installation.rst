.. _installation:

Installation
============

Install with pip
----------------
ScaleLLM is available as a Python Wheel package on `PyPI <https://pypi.org/project/scalellm/>`_. You can install it using pip:

.. code-block:: bash

	# Install scalellm with CUDA 12.1 and Pytorch 2.3
	$ pip install scalellm


Install other versions
----------------------
If you want to install ScaleLLM with different version of CUDA and Pytorch, you can pip install it with provding index URL of the version.

.. tabs::

	.. tab:: CUDA 12.1

		.. tabs::

			.. tab:: Pytorch 2.1

				.. code-block:: bash

					$ pip install scalellm -i https://whl.vectorch.com/cu121/torch2.1/

			.. tab:: Pytorch 2.2

				.. code-block:: bash

					$ pip install scalellm -i https://whl.vectorch.com/cu121/torch2.2/

			.. tab:: Pytorch 2.3

				.. code-block:: bash

					$ pip install scalellm -i https://whl.vectorch.com/cu121/torch2.3/

  	.. tab:: CUDA 11.8

		.. tabs::

			.. tab:: Pytorch 2.1

				.. code-block:: bash

					$ pip install scalellm -i https://whl.vectorch.com/cu118/torch2.1/

			.. tab:: Pytorch 2.2

				.. code-block:: bash

					$ pip install scalellm -i https://whl.vectorch.com/cu118/torch2.2/

			.. tab:: Pytorch 2.3

				.. code-block:: bash

					$ pip install scalellm -i https://whl.vectorch.com/cu118/torch2.3/



Build From Source
-----------------
If no wheel package is available for your configuration, you can build ScaleLLM from source code. You can clone the repository and install it locally using the following commands:

.. code-block:: bash
	
	$ git clone --recursive https://github.com/vectorch-ai/ScaleLLM.git
	$ cd ScaleLLM
	$ python3 setup.py bdist_wheel
	$ pip install dist/scalellm-*.whl


