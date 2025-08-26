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
import threading
import queue
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class TerminalManager:
    """Enhanced terminal manager with better resource management and features."""

    def __init__(self):
        self.terminals: Dict[str, Dict] = {}
        self._cleanup_lock = threading.Lock()

    def create_terminal(self, user_id: str) -> Dict:
        """Create a new terminal session for a user."""
        try:
            # Clean up existing terminal if any
            self.close_terminal(user_id)

            # Create pseudo-terminal
            main_fd, secondary_fd = pty.openpty()

            # Set terminal size and attributes
            self._set_terminal_size(main_fd, 24, 80)
            self._set_terminal_attributes(main_fd)

            # Start shell process
            env = os.environ.copy()
            env['TERM'] = 'xterm-256color'  # Better terminal type
            env['PS1'] = r'\u@\h:\w\$ '  # Standard prompt

            shell_process = subprocess.Popen(
                env.get('SHELL', '/bin/bash'),
                preexec_fn=os.setsid,
                stdin=secondary_fd,
                stdout=secondary_fd,
                stderr=secondary_fd,
                env=env,
                universal_newlines=False  # Handle binary data properly
            )

            # Close secondary fd - child process will use it
            os.close(secondary_fd)

            # Set non-blocking mode
            self._set_nonblocking(main_fd)

            # Initialize terminal data structure
            terminal_data = {
                'main_fd': main_fd,
                'process': shell_process,
                'pid': shell_process.pid,
                'created_at': time.time(),
                'last_activity': time.time(),
                'output_queue': queue.Queue(maxsize=1000),  # Buffer for output
                'input_buffer': '',  # Buffer for incomplete input sequences
                'command_history': [],  # Store command history
                'current_directory': os.getcwd(),
                'active': True
            }

            self.terminals[user_id] = terminal_data

            # Start output monitoring thread
            self._start_output_monitor(user_id)

            # Read initial output
            initial_output = self._read_with_timeout(main_fd, timeout=0.5)

            logger.info(f"Terminal created for user {user_id}")
            return {
                'success': True,
                'initial_output': initial_output,
                'terminal_id': user_id
            }

        except Exception as e:
            logger.error(f"Failed to create terminal for {user_id}: {e}")
            return {'success': False, 'error': str(e)}

    def _set_terminal_attributes(self, fd: int):
        """Set proper terminal attributes for better compatibility."""
        try:
            attrs = termios.tcgetattr(fd)
            # Enable canonical mode for better line editing
            attrs[3] |= termios.ECHO | termios.ICANON
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
        except (OSError, termios.error) as e:
            logger.warning(f"Could not set terminal attributes: {e}")

    def _set_nonblocking(self, fd: int):
        """Set file descriptor to non-blocking mode."""
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    def _set_terminal_size(self, fd: int, rows: int, cols: int):
        """Set terminal window size."""
        try:
            winsize = struct.pack("HHHH", rows, cols, 0, 0)
            fcntl.ioctl(fd, termios.TIOCSWINSZ, winsize)
        except OSError as e:
            logger.warning(f"Could not set terminal size: {e}")

    def _read_with_timeout(self, fd: int, timeout: float = 0.1) -> str:
        """Read from file descriptor with timeout."""
        try:
            ready, _, _ = select.select([fd], [], [], timeout)
            if ready:
                data = os.read(fd, 4096)
                return data.decode('utf-8', errors='replace')
        except (OSError, BlockingIOError):
            pass
        return ""

    def _start_output_monitor(self, user_id: str):
        """Start background thread to monitor terminal output."""
        def monitor():
            terminal = self.terminals.get(user_id)
            if not terminal:
                return

            fd = terminal['main_fd']
            output_queue = terminal['output_queue']

            while terminal.get('active', False):
                try:
                    output = self._read_with_timeout(fd, timeout=0.1)
                    if output:
                        terminal['last_activity'] = time.time()
                        if not output_queue.full():
                            output_queue.put(output)
                        else:
                            # Queue is full, drop oldest output
                            try:
                                output_queue.get_nowait()
                                output_queue.put(output)
                            except queue.Empty:
                                pass
                except Exception as e:
                    logger.error(f"Output monitor error for {user_id}: {e}")
                    break

                time.sleep(0.05)  # Small delay to prevent excessive CPU usage

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()

    def handle_input(self, user_id: str, data: str) -> Dict:
        """Handle terminal input with enhanced processing."""
        terminal = self.terminals.get(user_id)
        if not terminal or not terminal.get('active', False):
            return {'success': False, 'error': 'Terminal not found or inactive'}

        try:
            fd = terminal['main_fd']
            terminal['last_activity'] = time.time()

            # Handle special key sequences
            processed_data = self._process_input(data, terminal)

            # Write to terminal
            os.write(fd, processed_data.encode('utf-8'))

            # Update command history if it's a complete command
            if '\n' in data or '\r' in data:
                self._update_command_history(terminal, processed_data)

            return {'success': True}

        except Exception as e:
            logger.error(f"Input handling error for {user_id}: {e}")
            return {'success': False, 'error': str(e)}

    def _process_input(self, data: str, terminal: Dict) -> str:
        """Process input for special sequences and features."""
        # Handle tab completion
        if '\t' in data:
            return self._handle_tab_completion(data, terminal)

        # Handle Ctrl+C, Ctrl+D, etc.
        if len(data) == 1 and ord(data) < 32:
            return data  # Pass control characters through

        return data

    def _handle_tab_completion(self, data: str, terminal: Dict) -> str:
        """Handle tab completion by sending to shell."""
        # For now, pass tab through to shell which will handle completion
        # In the future, could implement custom completion logic
        return data

    def _update_command_history(self, terminal: Dict, command: str):
        """Update command history."""
        clean_command = command.strip()
        if clean_command and clean_command not in ['', '\n', '\r']:
            terminal['command_history'].append(clean_command)
            # Keep only last 1000 commands
            if len(terminal['command_history']) > 1000:
                terminal['command_history'] = terminal['command_history'][-1000:]

    def get_output(self, user_id: str) -> str:
        """Get pending output for a user."""
        terminal = self.terminals.get(user_id)
        if not terminal:
            return ""

        output_parts = []
        output_queue = terminal['output_queue']

        # Get all pending output
        while True:
            try:
                output_parts.append(output_queue.get_nowait())
            except queue.Empty:
                break

        return ''.join(output_parts)

    def resize_terminal(self, user_id: str, rows: int, cols: int) -> bool:
        """Resize terminal window."""
        terminal = self.terminals.get(user_id)
        if not terminal:
            return False

        try:
            self._set_terminal_size(terminal['main_fd'], rows, cols)
            logger.debug(f"Resized terminal for {user_id} to {rows}x{cols}")
            return True
        except Exception as e:
            logger.error(f"Resize error for {user_id}: {e}")
            return False

    def close_terminal(self, user_id: str):
        """Close a terminal session and cleanup resources."""
        with self._cleanup_lock:
            terminal = self.terminals.get(user_id)
            if not terminal:
                return

            # Mark as inactive to stop monitoring
            terminal['active'] = False

            try:
                # Terminate the process group
                os.killpg(os.getpgid(terminal['pid']), signal.SIGTERM)

                # Wait a bit for graceful termination
                time.sleep(0.1)

                # Force kill if still running
                try:
                    os.killpg(os.getpgid(terminal['pid']), signal.SIGKILL)
                except ProcessLookupError:
                    pass  # Already dead

            except (OSError, ProcessLookupError):
                pass  # Process already dead

            try:
                # Close file descriptor
                os.close(terminal['main_fd'])
            except OSError:
                pass

            # Remove from terminals dict
            del self.terminals[user_id]
            logger.info(f"Terminal closed for user {user_id}")

    def cleanup_inactive_terminals(self, max_idle_time: float = 3600):
        """Clean up terminals that have been inactive for too long."""
        current_time = time.time()
        inactive_users = []

        for user_id, terminal in self.terminals.items():
            if current_time - terminal['last_activity'] > max_idle_time:
                inactive_users.append(user_id)

        for user_id in inactive_users:
            logger.info(f"Cleaning up inactive terminal for {user_id}")
            self.close_terminal(user_id)

# Global terminal manager instance
terminal_manager = TerminalManager()

# Legacy function wrappers for compatibility
def create_terminal():
    """Legacy wrapper - creates terminal for 'default' user."""
    return terminal_manager.create_terminal('default')

def handle_terminal_input(terminal, data):
    """Legacy wrapper."""
    user_id = getattr(terminal, 'user_id', 'default')
    result = terminal_manager.handle_input(user_id, data.get('input', ''))
    if result['success']:
        return terminal_manager.get_output(user_id)
    return result.get('error', '')

def read_terminal_output(fd, max_read=4096):
    """Legacy wrapper."""
    # Find terminal by fd
    for user_id, terminal in terminal_manager.terminals.items():
        if terminal['main_fd'] == fd:
            return terminal_manager.get_output(user_id)
    return ""

def resize_terminal(terminal, rows, cols):
    """Legacy wrapper."""
    user_id = getattr(terminal, 'user_id', 'default')
    terminal_manager.resize_terminal(user_id, rows, cols)

def close_terminal(terminal):
    """Legacy wrapper."""
    user_id = getattr(terminal, 'user_id', 'default')
    terminal_manager.close_terminal(user_id)