#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import pty
import select
import subprocess
import fcntl
import termios
import struct
import signal
import time


def create_terminal():
    """
    Create a pseudo-terminal and initialize it with a shell process.

    Set up a pseudo-terminal by opening a new terminal device, configuring its size, launching a shell process,
    and returning the main file descriptor, the process ID of the shell, and any initial output from the shell.

    Returns
    -------
    dict
        A dictionary containing:
        - 'main': int
            The main file descriptor for the pseudo-terminal.
        - 'pid': int
            The process ID of the shell process.
        - 'last_output': str
            Any initial output from the shell.

    Raises
    ------
    OSError
        If opening a pseudo-terminal fails or if setting terminal size/attributes fails.
    """
    # Create a pseudo-terminal
    main, secondary = pty.openpty()

    # Set terminal size
    set_terminal_size(main, 24, 80)

    # Start a shell process
    shell = subprocess.Popen(
        os.environ.get('SHELL', '/bin/bash'),
        preexec_fn=os.setsid,
        stdin=secondary,
        stdout=secondary,
        stderr=secondary,
        universal_newlines=True
    )

    # Close secondary fd, we don't need it
    os.close(secondary)

    # Set non-blocking mode for main
    fl = fcntl.fcntl(main, fcntl.F_GETFL)
    fcntl.fcntl(main, fcntl.F_SETFL, fl | os.O_NONBLOCK)

    # Read the initial output
    time.sleep(0.1)  # Small delay to ensure output is ready
    initial_output = read_terminal_output(main)

    return {
        'main': main,
        'pid': shell.pid,
        'last_output': initial_output
    }


def set_terminal_size(fd, rows, cols):
    """
    Set the terminal size for the given file descriptor.

    Parameters
    ----------
    fd : int
        The file descriptor to set the terminal size for.
    rows : int
        The number of rows in the terminal.
    cols : int
        The number of columns in the terminal.
    """
    size = struct.pack("HHHH", rows, cols, 0, 0)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, size)


def read_terminal_output(fd, max_read=4096):
    """
    Reads output from a file descriptor and returns it as a string.

    This function reads data from the given file descriptor until there is no
    more data to read or the maximum number of bytes to read has been reached.
    The data is decoded using UTF-8 encoding, with any errors replaced by
    replacement characters. The function handles OSError and IOError exceptions
    silently.

    Parameters
    ----------
    fd : int
        The file descriptor from which to read the output.
    max_read : int, optional
        The maximum number of bytes to read at once (default is 4096).

    Returns
    -------
    str
        The output read from the file descriptor as a string.
    """
    output = ""
    try:
        while True:
            data = os.read(fd, max_read)
            if not data:
                break
            output += data.decode('utf-8', errors='replace')
    except (OSError, IOError):
        pass  # No more data to read
    return output


def handle_terminal_input(terminal, data):
    """
    Handle terminal input and retrieve output after a short delay.

    This function writes the given data to the specified terminal and retrieves any
    output generated as a result of that input. It is designed to handle terminal
    interactions where input is sent and output is expected to be read after a small
    delay.

    Parameters
    ----------
    terminal : dict
        A dictionary containing terminal-related information, including the 'main'
        key which refers to the file descriptor used for writing to and reading from
        the terminal.
    data : dict
        A dictionary containing data to be written to the terminal. The 'input' key
        is expected to contain the string data to be written.

    Returns
    -------
    str or None
        The output read from the terminal after writing the input, or an error message
        if an exception occurs during execution.
    """
    # Write input to the terminal
    try:
        input_data = data.get('input', '')
        os.write(terminal['main'], input_data.encode('utf-8'))

        # Wait a bit for output
        time.sleep(0.05)

        # Read output
        output = read_terminal_output(terminal['main'])
        return output
    except Exception as e:
        return f"\r\nError: {str(e)}\r\n"


def resize_terminal(terminal, rows, cols):
    """
    Resize the terminal to specified dimensions.

    This function resizes a terminal's viewport by setting its dimensions. It is commonly used
    to adjust the display area for better readability or to fit specific layout requirements.
    The function directly modifies the terminal object in-place and does not return any value.

    Parameters
    ----------
    terminal : dict
        A dictionary representing the terminal to be resized, containing a 'main' key with
        another dictionary as its value. This nested dictionary should contain terminal-specific
        configuration options, including viewport dimensions.
    rows : int
        The desired number of rows for the terminal's viewport. Should be a positive integer.
    cols : int
        The desired number of columns for the terminal's viewport. Should be a positive integer.

    See Also
    --------
    set_terminal_size: Function used internally to set the terminal dimensions.
    """
    set_terminal_size(terminal['main'], rows, cols)


def close_terminal(terminal):
    """
    Close a terminal session and its associated process.

    This function is used to close a terminal session by terminating its
    associated process and closing its file descriptors. It handles potential
    exceptions gracefully, such as when the process is already dead.

    Parameters
    ----------
    terminal : dict
        A dictionary containing terminal information with keys 'pid' and 'main'. The
        'pid' key holds the process ID of the associated process, and the 'main'
        key holds the file descriptor to be closed.
    """
    try:
        # Try to terminate the process gracefully
        os.killpg(os.getpgid(terminal['pid']), signal.SIGTERM)
        # Close the main fd
        os.close(terminal['main'])
    except:
        pass  # The Process might already be dead