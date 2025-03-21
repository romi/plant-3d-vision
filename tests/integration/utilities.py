import os
import subprocess

from romitask.log import get_logger

logger = get_logger(__name__)

def run_task(task, dataset, config=""):
    """Python wrapper to `romi_run_task` using subprocess.

    Parameters
    ----------
    task : str
        Name of the task to perform.
    dataset : str
        Patht to the dataset.
    config : str
        Path to the TOML configuration file.

    Returns
    -------
    subprocess.CompletedProcess
        The output of the subprocess.

    """
    # Check if PYOPENCL_CTX is set
    if os.getenv('PYOPENCL_CTX') == None:
       os.environ["PYOPENCL_CTX"] = '0'

    command = ["romi_run_task", "--config", config, task, dataset]
    process = subprocess.run(command, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(process.stdout)
    print(process.stderr)

    return process