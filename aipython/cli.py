"""Command line interface for aipython."""

import sys
from IPython import start_ipython


def main():
    """Launch IPython with aipython extension pre-loaded."""
    # Add our extension to the startup
    sys.argv.extend(['--ext', 'aipython'])
    start_ipython()


if __name__ == '__main__':
    main()
