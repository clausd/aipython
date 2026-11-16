# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""`aipython` command: IPython with the extension preloaded."""

import sys

from IPython import start_ipython


def main():
    start_ipython(argv=["--ext", "aipython", *sys.argv[1:]])


if __name__ == "__main__":
    main()
