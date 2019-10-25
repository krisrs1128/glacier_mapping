#!/usr/bin/env python
from pathlib import Path


def env_to_path(path):
    """Transorms an environment variable mention in a conf file
    into its actual value. E.g. $HOME/clouds -> /home/vsch/clouds

    Args:
        path (str): path potentially containing the env variable
    """
    if not isinstance(path, str):
        return path

    path_elements = path.split("/")
    for i, d in enumerate(path_elements):
        if "$" in d:
            path_elements[i] = os.environ.get(d.replace("$", ""))
    if any(d is None for d in path_elements):
        return ""
    return "/".join(path_elements)

