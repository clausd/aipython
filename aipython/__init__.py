# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Seamless AI for IPython: type Python or describe what you want."""

__version__ = "0.2.0"

_installed = None


def load_ipython_extension(ipython):
    global _installed
    from .shell import AIShell
    if _installed is None:
        _installed = AIShell(ipython)
        _installed.install()


def unload_ipython_extension(ipython):
    global _installed
    if _installed is not None:
        _installed.uninstall()
        _installed = None
