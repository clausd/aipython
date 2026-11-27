# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Missing packages: say which package to install, and install it on request.

aipython is often installed with `uv tool install`, in an environment of its own that has no
pip, so `%pip install` fails there. `%ai install` uses uv when it is available and pip otherwise,
installing into the environment aipython is running in.
"""

import importlib
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

# Modules whose package on PyPI has a different name.
PACKAGES = {
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "dotenv": "python-dotenv",
}


def package_for(module):
    top = module.split(".")[0]
    return PACKAGES.get(top, top)


def is_uv_tool(prefix=None):
    return (Path(prefix or sys.prefix) / "uv-receipt.toml").exists()


def install_command(packages):
    uv = shutil.which("uv")
    if uv:
        return [uv, "pip", "install", "--python", sys.executable, *packages]
    return [sys.executable, "-m", "pip", "install", *packages]


def hint(error):
    """What to tell the user when a cell failed because a module isn't installed, else None."""
    if not isinstance(error, ModuleNotFoundError) or not error.name:
        return None
    top = error.name.split(".")[0]
    try:
        if importlib.util.find_spec(top) is not None:  # a missing submodule, not a package
            return None
    except (ImportError, ValueError):
        pass
    package = package_for(top)
    text = f"aipython: {top} isn't installed here. Install it with:  %ai install {package}"
    if is_uv_tool():
        text += f"\n   to keep it for good:  uv tool install aipython-shell --with {package}"
    return text


def install(packages, run=subprocess.run):
    """Install `packages` into the running environment; return True on success."""
    command = install_command(packages)
    print("$ " + " ".join(command))
    if run(command).returncode != 0:
        return False
    importlib.invalidate_caches()
    return True
