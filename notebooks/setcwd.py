"""Script that sets up the file directory tree before running a notebook.

It uses root of github directory to make sure everyone's code runs from
the same directory, called current working directory cwd.

It adds the python code in the parent directory of the working directory
in the list of paths.

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
    )

    print("Git root path: ", gitroot_path) # /home/adele/code/gagf-agf/

    os.chdir(os.path.join(gitroot_path[:-1], "gagf"))
    print("Working directory: ", os.getcwd()) # /home/adele/code/group-agf/gagf-agf

    sys_dir = os.path.dirname(os.getcwd())
    sys.path.append(sys_dir) # /home/adele/code/group-agf
    print("Directory added to path: ", sys_dir) 
    notebook_dir = os.path.join(os.getcwd(), "notebooks")
    sys.path.append(notebook_dir)
    print("Directory added to path: ", notebook_dir)


def get_root_dir():
    """Return the root directory of the git repository."""
    gitroot_path = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], universal_newlines=True
    )
    return gitroot_path[:-1]  # Remove trailing newline
    



