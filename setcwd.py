"""Script that sets up the file directory tree before running a notebook.

It uses root of github directory to make sure everyone's code runs from
the same directory, called current working directory cwd.

It adds the repo root to sys.path so imports work correctly.

Usage:

import setcwd
setcwd.main()

"""

import os
import subprocess
import sys
import warnings


def main():
    """Set up the file paths and directory tree."""
    warnings.filterwarnings("ignore")

    gitroot_path = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], universal_newlines=True
    ).strip()

    os.chdir(gitroot_path)
    print("Working directory: ", os.getcwd())

    if gitroot_path not in sys.path:
        sys.path.insert(0, gitroot_path)
    print("Directory added to path: ", gitroot_path)


def get_root_dir():
    """Return the root directory of the git repository."""
    return subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], universal_newlines=True
    ).strip()
    



