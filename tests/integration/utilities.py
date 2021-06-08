import os
import subprocess

def run_task(config, task, data):
    # Check if PYOPENCL_CTX is set
    if os.getenv('PYOPENCL_CTX') == None:
       os.environ["PYOPENCL_CTX"] = '0'

    command = ["romi_run_task", "--config", config, task, data]
    process = subprocess.run(command, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(process.stdout)
    print(process.stderr)

    return process