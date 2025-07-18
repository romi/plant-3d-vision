#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#  Copyright (c) 2022 Univ. Lyon, ENS de Lyon, UCB Lyon 1, CNRS, INRAe, Inria
#  All rights reserved.
#  This file is part of the TimageTK library, and is released under the "GPLv3"
#  license. Please see the LICENSE.md file that should have been included as
#  part of this package.
# ------------------------------------------------------------------------------

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
    """Create a new terminal."""
    # Create a pseudo-terminal
    master, slave = pty.openpty()

    # Set terminal size
    set_terminal_size(master, 24, 80)

    # Start a shell process
    shell = subprocess.Popen(
        os.environ.get('SHELL', '/bin/bash'),
        preexec_fn=os.setsid,
        stdin=slave,
        stdout=slave,
        stderr=slave,
        universal_newlines=True
    )

    # Close slave fd, we don't need it
    os.close(slave)

    # Set non-blocking mode for master
    fl = fcntl.fcntl(master, fcntl.F_GETFL)
    fcntl.fcntl(master, fcntl.F_SETFL, fl | os.O_NONBLOCK)

    # Read initial output
    time.sleep(0.1)  # Small delay to ensure output is ready
    initial_output = read_terminal_output(master)

    return {
        'master': master,
        'pid': shell.pid,
        'last_output': initial_output
    }


def set_terminal_size(fd, rows, cols):
    """Set terminal size."""
    size = struct.pack("HHHH", rows, cols, 0, 0)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, size)


def read_terminal_output(fd, max_read=4096):
    """Read output from the terminal."""
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
    """Handle input to the terminal and return output."""
    # Write input to terminal
    try:
        input_data = data.get('input', '')
        os.write(terminal['master'], input_data.encode('utf-8'))

        # Wait a bit for output
        time.sleep(0.05)

        # Read output
        output = read_terminal_output(terminal['master'])
        return output
    except Exception as e:
        return f"\r\nError: {str(e)}\r\n"


def resize_terminal(terminal, rows, cols):
    """Resize the terminal."""
    set_terminal_size(terminal['master'], rows, cols)


def close_terminal(terminal):
    """Close the terminal."""
    try:
        # Try to terminate the process gracefully
        os.killpg(os.getpgid(terminal['pid']), signal.SIGTERM)
        # Close the master fd
        os.close(terminal['master'])
    except:
        pass  # The Process might already be dead