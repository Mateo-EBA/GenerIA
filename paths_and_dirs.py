#%% --- --- --- --- --- --- --- --- ---
# Imports
import os
import glob
import shutil
from pathlib import Path

#%% --- --- --- --- --- --- --- --- ---
# Paths
def add_default_extension(path:str, extension:str):
    """
    Adds a default extension to a path if it does not already have one.

    Args:
        path (str): The path to check.
        extension (str): The default extension to add.

    Returns:
        str: The path with the default extension added.
    """
    _, ext = os.path.splitext(path)
    if not ext:
        return f'{path}.{extension}'
    else:
        return path

def get_files_from_dir(directory:str, extension:str=None, include_subdirs:bool=True) -> list[str]:
    """
    Get a list of file paths from a directory matching a given extension.

    Args:
        directory (str): The directory to search for files.
        extension (str, optional): The file extension to search for. Defaults to None.
        include_subdirs (bool, optional): Whether to include files from subdirectories. Defaults to True.

    Returns:
        list[str]: A list of file paths.
    """
    if extension is None:
        extension = "*"
    else:
        extension = extension.replace(".", "")
        
    if not include_subdirs:
        file_paths = glob.glob(os.path.join(directory, f'*.{extension}'))
    else:
        file_paths = []
        for path in Path(directory).rglob(f'*.{extension}'):
            file_paths.append(str(path))
    return file_paths

def copy_directory(src_dir:str, dst_root:str):
    """
    Copies a directory and all its contents to a new root.

    Args:
        src_dir (str): The source directory to be copied.
        dst_dir (str): The destination directory where the copied directory will be placed.

    Returns:
        None
    """
    # Create the destination directory if it doesn't exist
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)
    # Copy the directory and all its contents to the destination
    shutil.copytree(src_dir, os.path.join(dst_root, os.path.basename(src_dir)))
