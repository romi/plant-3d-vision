#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Web Terminal Interface with Authentication

A Flask-based web application that provides a secure, interactive terminal interface accessible through a web browser.
It enables remote terminal access with user authentication, session management, and real-time interaction using WebSocket technology.

## Key Features

- Secure user authentication and session management
- Real-time terminal interaction via WebSocket (Socket.IO)
- User profile management with password change functionality
- Admin interface for user management
- Support for reverse proxy configuration
- Automatic terminal cleanup for inactive sessions
- API endpoints for scan management and TOML file operations
- Configurable logging and debugging options
- Environment variable support for flexible deployment

## Usage Examples

>>> # Start the server with default settings
>>> webterm_server()

>>> # Start with custom configuration
>>> webterm_server(
...     host='localhost',
...     port=5000,
...     proxy=True,
...     debug=True,
...     users_db_path='custom_users.csv',
...     log_level='DEBUG'
... )

>>> # Using command line interface
>>> # python app.py --host localhost --port 5000 --proxy --debug --log-level DEBUG
"""

import argparse
import os
import threading
import time

from dotenv import load_dotenv
from flask import Flask
from flask import jsonify
from flask import redirect
from flask import render_template
from flask import request
from flask import session
from flask import url_for
from flask_socketio import SocketIO
from werkzeug.middleware.proxy_fix import ProxyFix

from plant3dvision.webterm.auth import authenticate_user
from plant3dvision.webterm.auth import format_csv_line
from plant3dvision.webterm.auth import hash_password
from plant3dvision.webterm.auth import load_users
from plant3dvision.webterm.terminal import terminal_manager
from plantdb.commons.fsdb import FSDB
from romitask.log import DEFAULT_LOG_LEVEL
from romitask.log import LOG_LEVELS
from romitask.log import get_logger

# Load environment variables from .env file
load_dotenv(verbose=False, override=True)


def create_webterm_app(proxy=False, users_db_path='users.csv', secret_key=None,
                       log_level=DEFAULT_LOG_LEVEL, async_mode='threading'):
    """Create and configure the WebTerm Flask application.

    Parameters
    ----------
    proxy : bool, optional
        Boolean flag indicating whether the application is behind a reverse proxy, by default False
    users_db_path : str, optional
        Path to the users database CSV file, by default 'users.csv'
    secret_key : str, optional
        Secret key for session management, by default None
    log_level : str, optional
        Logging level, by default DEFAULT_LOG_LEVEL

    Returns
    -------
    SocketIO
        Configured SocketIO application instance
    """
    # Get the directory where this app.py file is located
    app_dir = os.path.dirname(os.path.abspath(__file__))

    # Initialize the Flask application with explicit template and static folders
    app = Flask("WebTerm",
                template_folder=os.path.join(app_dir, 'templates'),
                static_folder=os.path.join(app_dir, 'static'))

    logger = get_logger("WebTerm", log_level=log_level)

    if not secret_key:
        logger.warning("No secret key found, using a random key.")
        logger.warning("Please set the SERVER_SECRET_KEY environment variable.")
        secret_key = os.urandom(24)

    app.secret_key = secret_key

    # Configure proxy settings if needed
    if proxy:
        logger.info(f"Setting up Flask application with proxy support...")
        prefix = os.environ.get("WEBTERM_PREFIX", "")
        logger.info(f"Using prefix '{prefix}' for all endpoints.")
        # App is behind one proxy that sets the -For and -Host headers.
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1, x_proto=1)
        # Set secure cookies
        app.config.update(
            SESSION_COOKIE_SECURE=True,
            SESSION_COOKIE_SAMESITE='Lax'
        )

    # Initialize Socket.IO server with better configuration
    socketio_config = {
        'async_mode': async_mode,
        'cors_allowed_origins': "*",
        'ping_timeout': 60,
        'ping_interval': 25,
        'logger': False,  # Disable socketio logging to avoid conflicts with WSGI
        'engineio_logger': False
    }

    if proxy:
        # Additional SocketIO configuration for proxy environments
        socketio_config.update({
            'cors_allowed_origins': "*",
            'allow_credentials': True
        })

    socketio = SocketIO(app, **socketio_config)

    # Background cleanup task
    def start_cleanup_task():
        """Start the terminal cleanup background task."""

        def cleanup_inactive_terminals():
            """Background task to clean up inactive terminals."""
            while True:
                try:
                    # Clean up terminals inactive for more than 1 hour
                    terminal_manager.cleanup_inactive_terminals(max_idle_time=3600)
                    time.sleep(300)  # Run cleanup every 5 minutes
                except Exception as e:
                    logger.error(f"Cleanup task error: {e}")
                    time.sleep(60)  # Wait longer on error

        # Start cleanup task
        cleanup_thread = threading.Thread(target=cleanup_inactive_terminals, daemon=True)
        cleanup_thread.start()
        logger.info("Terminal cleanup task started")

    # Socket.IO event handlers with enhanced error handling
    @socketio.on('connect')
    def handle_connect(auth=None):
        if 'username' not in session:
            logger.warning("Connection rejected: No valid session")
            return False

        username = session.get('username')
        logger.info(f"User {username} connected")

        try:
            # Create terminal for user if it doesn't exist
            if username not in terminal_manager.terminals:
                result = terminal_manager.create_terminal(username)
                if not result.get('success', False):
                    logger.error(f"Failed to create terminal for {username}: {result.get('error')}")
                    return False

                # Change to $ROMI_DB directory automatically
                terminal_manager.handle_input(username, 'cd $ROMI_DB\n')

            return True
        except Exception as e:
            logger.error(f"Connection error for {username}: {e}")
            return False

    @socketio.on('disconnect')
    def handle_disconnect():
        username = session.get('username')
        if username:
            logger.info(f"User {username} disconnected")
            # Don't close terminal on disconnect - preserve state for reconnection

    @socketio.on('resize')
    def handle_resize(data):
        """Handle terminal resize events from the client."""
        username = session.get('username')
        if not username:
            return

        try:
            rows = max(1, min(100, data.get('rows', 24)))  # Validate dimensions
            cols = max(1, min(300, data.get('cols', 80)))

            success = terminal_manager.resize_terminal(username, rows, cols)
            if not success:
                logger.warning(f"Failed to resize terminal for {username}")
        except Exception as e:
            logger.error(f"Resize error for {username}: {e}")

    @socketio.on('terminal_input')
    def socket_handle_terminal_input(data):
        username = session.get('username')
        if not username:
            return

        try:
            input_data = data.get('input', '')
            if not input_data:
                return

            result = terminal_manager.handle_input(username, input_data)
            if not result.get('success', False):
                logger.warning(f"Input handling failed for {username}: {result.get('error')}")
        except Exception as e:
            logger.error(f"Terminal input error for {username}: {e}")
            socketio.emit('terminal_output', {
                'output': f'\r\n\x1b[31mError processing input: {str(e)}\x1b[0m\r\n'
            }, room=request.sid)

    @socketio.on('start_output_polling')
    def start_output_polling():
        username = session.get('username')
        if not username or username not in terminal_manager.terminals:
            return

        sid = request.sid
        terminal = terminal_manager.terminals[username]

        # Mark polling as active for this session
        terminal['polling_active'] = True
        terminal['current_sid'] = sid

        def poll_output():
            while terminal.get('polling_active', False) and terminal.get('current_sid') == sid:
                try:
                    output = terminal_manager.get_output(username)
                    if output:
                        socketio.emit('terminal_output', {'output': output}, room=sid)
                except Exception as e:
                    logger.error(f"Output polling error for {username}: {e}")
                    break

                # Use eventlet sleep to be cooperative
                socketio.sleep(0.05)

        # Start polling in a background task
        socketio.start_background_task(poll_output)

    @socketio.on('stop_output_polling')
    def stop_output_polling():
        username = session.get('username')
        if username and username in terminal_manager.terminals:
            terminal_manager.terminals[username]['polling_active'] = False

    @app.route('/')
    def index():
        if 'username' in session:
            return redirect(url_for('terminal'))
        return render_template('login.html')

    @app.route('/login', methods=['POST'])
    def login():
        username = request.form.get('username')
        password = request.form.get('password')

        user = authenticate_user(username, password)
        if user:
            session['username'] = username
            session['full_name'] = user['full_name']
            logger.info(f"User {username} logged in successfully")
            return redirect(url_for('terminal'))

        logger.warning(f"Failed login attempt for username: {username}")
        return render_template('login.html', error='Invalid credentials')

    @app.route('/logout')
    def logout():
        username = session.get('username')
        if username:
            # Close the terminal when user logs out
            terminal_manager.close_terminal(username)
            logger.info(f"User {username} logged out")

        session.clear()
        return redirect(url_for('index'))

    @app.route('/terminal')
    def terminal():
        if 'username' not in session:
            return redirect(url_for('index'))
        return render_template('terminal.html',
                               full_name=session.get('full_name'),
                               username=session.get('username'))

    @app.route('/admin')
    def admin_panel():
        if 'username' not in session or session.get('username') != 'admin':
            return redirect(url_for('index'))
        return render_template('admin.html')

    @app.route('/admin/add_user', methods=['POST'])
    def add_user():
        # Simple admin endpoint to add users
        if session.get('username') != 'admin':  # Basic admin check
            return {'success': False, 'error': 'Unauthorized'}, 403

        try:
            full_name = request.form.get('full_name')
            username = request.form.get('username')
            password = request.form.get('password')

            if not all([full_name, username, password]):
                return {'success': False, 'error': 'Missing required fields'}, 400

            # Add user to CSV
            with open(users_db_path, 'a') as f:
                password_hash = hash_password(password)
                f.write(format_csv_line(full_name, username, password_hash))

            return {'success': True}, 200
        except Exception as e:
            return {'success': False, 'error': str(e)}, 500

    @app.route('/api/scans', methods=['GET'])
    def get_scans():
        try:
            db = FSDB(os.getenv('ROMI_DB', '/myapp/db'))
            db.connect(unsafe=True)
            list_scan_names = db.list_scans(owner_only=False)
            db.disconnect()
            return jsonify(list_scan_names)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/list-toml-files', methods=['GET'])
    def list_toml_files():
        username = request.args.get('username', 'default')

        # Get the directory from environment variable or use default
        default_path = f'/myapp/cfg/{username}/'
        save_dir = os.environ.get('ROMI_CFG', default_path)

        try:
            # Ensure directory exists
            os.makedirs(save_dir, exist_ok=True)

            # Get all TOML files in the directory
            files = [f for f in os.listdir(save_dir) if f.lower().endswith('.toml')]

            return jsonify({
                'success': True,
                'files': files,
                'directory': save_dir
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/load-toml-file', methods=['GET'])
    def load_toml_file():
        filename = request.args.get('filename')
        username = request.args.get('username', 'default')

        if not filename:
            return jsonify({
                'success': False,
                'error': 'Filename is required'
            }), 400

        # Get the directory from environment variable or use default
        default_path = f'/myapp/cfg/{username}/'
        save_dir = os.environ.get('ROMI_CFG', default_path)

        try:
            # Construct a full file path
            file_path = os.path.join(save_dir, filename)

            # Check if file exists
            if not os.path.exists(file_path):
                return jsonify({
                    'success': False,
                    'error': f'File not found: {filename}'
                }), 404

            # Read file content
            with open(file_path, 'r') as f:
                content = f.read()

            return jsonify({
                'success': True,
                'filename': filename,
                'content': content,
                'path': file_path
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/save-toml', methods=['POST'])
    def save_toml():
        data = request.json
        filename = data.get('filename')
        content = data.get('content')

        # Get username from session or request
        username = session.get('username')  # Assuming username is stored in session

        # Get the save directory from environment variable or use default
        if username:
            default_path = f'/myapp/cfg/{username}/'
        else:
            default_path = '/myapp/cfg/'  # Fallback path if username is not available

        save_dir = os.environ.get('ROMI_CFG', default_path)

        # Ensure directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Save the file
        file_path = os.path.join(save_dir, filename)

        try:
            with open(file_path, 'w') as f:
                f.write(content)
            return jsonify({
                'success': True,
                'path': file_path,
                'message': f'File saved to {save_dir}{filename}'
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/user/profile')
    def user_profile():
        if 'username' not in session:
            return redirect(url_for('index'))
        return render_template('user_profile.html',
                               full_name=session.get('full_name'),
                               username=session.get('username'))

    @app.route('/user/change_password', methods=['POST'])
    def change_password():
        if 'username' not in session:
            return {'success': False, 'error': 'Unauthorized'}, 403

        try:
            current_password = request.form.get('current_password')
            new_password = request.form.get('new_password')

            if not all([current_password, new_password]):
                return {'success': False, 'error': 'Missing required fields'}, 400

            # Verify the current password
            username = session.get('username')
            users = load_users()

            if username not in users:
                return {'success': False, 'error': 'User not found'}, 404

            # Verify the current password
            from auth import verify_password
            if not verify_password(users[username]['password_hash'], current_password):
                return {'success': False, 'error': 'Current password is incorrect'}, 401

            # Update password in CSV
            from auth import hash_password
            new_password_hash = hash_password(new_password)
            users[username]['password_hash'] = new_password_hash

            # Write all users back to CSV
            with open(users_db_path, 'w') as f:
                f.write('"full_name";"username";"password_hash"\n')  # Header
                for user, data in users.items():
                    f.write(format_csv_line(data['full_name'], user, data['password_hash']))

            return {'success': True}, 200
        except Exception as e:
            return {'success': False, 'error': str(e)}, 500

    # Error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return render_template('error.html', error_code=404, error_message="Page not found"), 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        return render_template('error.html', error_code=500, error_message="Internal server error"), 500

    # Create users.csv if it doesn't exist
    if not os.path.exists(users_db_path):
        logger.warning("No existing users database found, creating a new one.")
        with open(users_db_path, 'w') as f:
            f.write(format_csv_line("full_name", "username", "password_hash"))
            # Add default admin user
            admin_hash = hash_password('admin')
            f.write(format_csv_line("Administrator", "admin", admin_hash))
    else:
        logger.info("Existing users database found.")

    # Start background tasks
    start_cleanup_task()

    logger.info("WebTerm application created successfully")
    return socketio, app


def parsing():
    """Parse command line arguments for the WebTerm server."""
    parser = argparse.ArgumentParser(description='WebTerm - Web-based terminal interface with authentication.')

    server_args = parser.add_argument_group("webserver arguments")
    server_args.add_argument('--host', type=str, default=os.environ.get('SERVER_HOST', '0.0.0.0'),
                             help="the hostname to listen on, defaults to '0.0.0.0'.")
    server_args.add_argument('--port', type=int, default=int(os.environ.get('SERVER_PORT', 8080)),
                             help="the port of the webserver, defaults to '8080'.")
    server_args.add_argument('--proxy', action='store_true',
                             help="use this flag when this server sits behind a reverse proxy")
    server_args.add_argument('--debug', action='store_true',
                             help="enable debug mode.")

    auth_args = parser.add_argument_group("authentication arguments")
    auth_args.add_argument('--users-db', dest='users_db_path', type=str,
                           default=os.environ.get('USERS_DB_PATH', 'users.csv'),
                           help="path to the users database CSV file, defaults to 'users.csv'.")
    auth_args.add_argument('--secret-key', dest='secret_key', type=str,
                           default=os.environ.get('SERVER_SECRET_KEY'),
                           help="secret key for session management.")

    log_opt = parser.add_argument_group("logging options")
    log_opt.add_argument("--log-level", dest="log_level", type=str, default=DEFAULT_LOG_LEVEL, choices=LOG_LEVELS,
                         help="level of message logging, defaults to 'INFO'.")

    return parser


def webterm_server(host='0.0.0.0', port=8080, proxy=False, debug=False, users_db_path='users.csv', secret_key=None,
                   log_level=DEFAULT_LOG_LEVEL):
    """Initialize and start the WebTerm server.

    Parameters
    ----------
    host : str, optional
        The hostname to listen on, by default '0.0.0.0'
    port : int, optional
        The port of the webserver, by default 8080
    proxy : bool, optional
        Boolean flag indicating whether the application is behind a reverse proxy, by default False
    debug : bool, optional
        Enable debug mode, by default False
    users_db_path : str, optional
        Path to the users database CSV file, by default 'users.csv'
    secret_key : str, optional
        Secret key for session management, by default None
    log_level : str, optional
        Logging level, by default DEFAULT_LOG_LEVEL
    """
    # Create the application using the factory function
    socketio_app, flask_app = create_webterm_app(
        proxy=proxy,
        users_db_path=users_db_path,
        secret_key=secret_key,
        log_level=log_level,
    )

    # Start the server
    socketio_app.run(flask_app, host=host, port=port, debug=debug)


def main():
    """Main function to initialize and execute the WebTerm server.

    This function utilizes argument parsing to extract user-provided input values
    for configuring and running the WebTerm server.
    """
    parser = parsing()
    args = parser.parse_args()

    webterm_server(
        host=args.host,
        port=args.port,
        proxy=args.proxy,
        debug=args.debug,
        users_db_path=args.users_db_path,
        secret_key=args.secret_key,
        log_level=args.log_level
    )


if __name__ == '__main__':
    main()
