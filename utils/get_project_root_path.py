import os
import os.path as path


def get_project_root_path():
    return path.abspath(path.join(__file__, '..', '..'))
