import os


def get_version_suffix():
    try:
        import torch

        version, other = torch.__version__.split("+")
        major, minor, _ = version.split(".")
        return f"{other}torch{major}.{minor}"
    except ImportError:
        return None


def rename_whl(whl_path: str):
    version_suffix = get_version_suffix()
    if version_suffix is None:
        return

    parts = whl_path.split("-")
    if len(parts) < 2:
        return
    version = parts[1]
    # check if already added version suffix
    if version.endswith(version_suffix):
        return

    parts[1] = f"{version}+{version_suffix}"
    new_whl_path = "-".join(parts)
    os.rename(whl_path, new_whl_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python rename_whl.py <whl_path>")
        sys.exit(1)
    whl_path = sys.argv[1]
    if not os.path.exists(whl_path):
        print(f"File not found: {whl_path}")
        sys.exit(1)
    rename_whl(whl_path)
