import os
import subprocess
import sys
import time

import requests


class OpenAIServer:
    def __init__(self, args):
        env = os.environ.copy()
        # convert all args to string
        args = [str(arg) for arg in args]
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "scalellm.serve.api_server", *args],
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

    def wait_ready(self, timeout):
        start = time.time()
        while True:
            try:
                if requests.get("http://localhost:8080/health").status_code == 200:
                    break
            except Exception as e:
                # check if process is still running
                if self._proc.poll() is not None:
                    raise RuntimeError("Server process exited") from e

                # wait a bit before retry
                now = time.time()
                if now - start > timeout:
                    raise TimeoutError("Server not ready in time") from e
                else:
                    time.sleep(0.5)

    def __del__(self):
        if hasattr(self, "_proc"):
            self._proc.terminate()
            self._proc.wait()

    def __enter__(self):
        # wait up to 60 seconds for the server to start
        self.wait_ready(60)
        return self

    def __exit__(self, *args):
        self.__del__()


if __name__ == "__main__":
    with OpenAIServer(sys.argv[1:]) as server:
        input("Press Enter to stop server...")
