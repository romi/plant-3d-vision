#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import subprocess

import pycuda.autoinit
import pycuda.compiler
import pycuda.driver as cuda

from romitask.log import get_logger

logger = get_logger(__name__)

# Mapping of CUDA versions to maximum supported SM architecture
CUDA_VERSION_TO_MAX_ARCH = {
    # CUDA 11.x
    (11, 0): '80',  # CUDA 11.0-11.4 support up to sm_80
    (11, 5): '87',  # CUDA 11.5-11.8 support up to sm_87
    (11, 8): '87',
    # CUDA 12.x
    (12, 0): '89',  # CUDA 12.0-12.4 support up to sm_89
    (12, 4): '89',
    (12, 5): '90',  # CUDA 12.5+ support up to sm_90
    # CUDA 13.x
    (13, 0): '90',  # CUDA 13.0+ support up to sm_90
}


def get_nvcc_version() -> tuple[int, int] | None:
    """
    Get the NVCC compiler version.

    Returns the major and minor version numbers of the NVCC compiler.
    If NVCC is not found, attempts to retrieve the CUDA version as a fallback.
    Returns ``None`` if neither NVCC nor CUDA is available.

    Returns
    -------
    tuple[int, int] or None
        A tuple containing the major and minor version numbers (e.g., ``(11, 6)``),
        or ``None`` if the version cannot be determined.

    Raises
    ------
    FileNotFoundError
        If the ``nvcc`` executable is not found in the system path.
    subprocess.CalledProcessError
        If ``nvcc`` execution fails for an unexpected reason.

    Notes
    -----
    This function first tries to query the NVCC compiler version. If that fails,
    it falls back to querying the CUDA version. The returned tuple corresponds
    to the major and minor version numbers (e.g., ``(11, 6)`` for NVCC 11.6).
    """
    try:
        output = subprocess.check_output(['nvcc', '--version'],
                                         stderr=subprocess.STDOUT,
                                         text=True)
        # Look for pattern like "release 11.5"
        match = re.search(r'release\s+(\d+)\.(\d+)', output)
        if match:
            return int(match.group(1)), int(match.group(2))
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return cuda.get_version()[:2]


def get_max_supported_arch(nvcc_version=None) -> str:
    """
    Get the maximum supported SM architecture for the NVCC version.

    Determines the highest compute capability (SM architecture) supported by the given NVCC version.
    If no version is provided, attempts to auto-detect it. Falls back to conservative defaults
    for very old or undetectable CUDA versions.

    Parameters
    ----------
    nvcc_version : tuple[int, int] | None, optional
        CUDA version as a tuple of (major, minor) numbers. If ``None`` (default),
        the version is auto-detected using :func:`get_nvcc_version`.

    Returns
    -------
    str
        String representing the maximum supported architecture (e.g., ``'87'``, ``'89'``, ``'90'``).

    Notes
    -----
    - If auto-detection fails, a warning is printed and ``'60'`` is returned.
    - For very old CUDA versions, a warning is printed and ``'60'`` is returned.

    See Also
    --------
    get_nvcc_version : Function used to auto-detect the CUDA version.
    """
    if nvcc_version is None:
        nvcc_version = get_nvcc_version()

    if nvcc_version is None:
        # Fallback to a conservative default
        logger.warning("Could not detect NVCC version, using conservative default 'sm_60'")
        return '60'

    major, minor = nvcc_version

    # Find the best matching version
    for (v_major, v_minor), arch in sorted(CUDA_VERSION_TO_MAX_ARCH.items(), reverse=True):
        if major > v_major or (major == v_major and minor >= v_minor):
            return arch

    # If version is older than what we know, use the oldest supported
    logger.warning(f"CUDA {major}.{minor} is very old, using conservative default 'sm_60'")
    return '60'


def get_capped_arch(nvcc_version=None) -> str:
    """
    Get the architecture to use, capped at the maximum supported by NVCC.

    This function determines the appropriate CUDA architecture string for JIT compilation,
    ensuring it does not exceed the maximum supported by the NVCC compiler version.

    Parameters
    ----------
    nvcc_version : tuple[int, int] | None, optional
        A tuple specifying the NVCC major and minor version (e.g., (11, 8)).
        If ``None`` (default), the version is auto-detected from the system.

    Returns
    -------
    str
        Architecture string for SourceModule (e.g., ``'sm_75'`` or ``'sm_89'``).

    Raises
    ------
    RuntimeError
        If the CUDA device cannot be accessed or the compute capability is unsupported.

    Notes
    -----
    - If the detected device architecture exceeds the NVCC maximum, a warning is printed
      and the capped architecture is used instead.
    """
    max_arch = get_max_supported_arch(nvcc_version)

    device = cuda.Device(0)
    major, minor = device.compute_capability()
    device_arch_num = int(f'{major}{minor}')
    max_arch_num = int(max_arch)

    if device_arch_num > max_arch_num:
        # Use virtual architecture for JIT compilation
        arch = f'sm_{max_arch}'
        logger.warning(f"Device sm_{major}{minor} exceeds NVCC max sm_{max_arch}")
    else:
        # Use device's actual architecture
        arch = f'sm_{major}{minor}'

    logger.info(f"Using {arch} for JIT compilation at runtime")
    return arch


if __name__ == "__main__":
    # Auto-detect NVCC version
    nvcc_version = get_nvcc_version()
    print(f"NVCC Version: {nvcc_version}")

    max_arch = get_max_supported_arch(nvcc_version)
    print(f"Maximum supported architecture: sm_{max_arch}")

    # Get the capped architecture to use
    arch_to_use = get_capped_arch(nvcc_version)
    print(f"Architecture to use: {arch_to_use}")

    # Compile with the capped architecture
    code = """
    __global__ void kernel() { }
    """
    try:
        module = pycuda.compiler.SourceModule(code, arch=arch_to_use)
    except RuntimeError as e:
        print(f"Detected architecture failed to compile!")
        raise e
    else:
        print(f"Detected architecture compiled successfully!")
