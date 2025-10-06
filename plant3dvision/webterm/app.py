#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Web Terminal Interface with Authentication

A Flask-based web application that provides a secure, interactive terminal interface accessible through a web browser.
It enables remote terminal access with user authentication, session management, and real-time interaction using WebSocket technology.

Key Features
------------
- Secure user authentication and session management
- Real-time terminal interaction via WebSocket (Socket.IO)
- User profile management with password change functionality
- Admin interface for user management
- Support for reverse proxy configuration
- Automatic terminal cleanup for inactive sessions
- API endpoints for scan management and TOML file operations
- Configurable logging and debugging options
- Environment variable support for flexible deployment

Usage Examples
--------------
Using Python API:

.. code-block:: python
   :linenos:

   # Start the server with default settings
   webterm_server()
   # Start with custom configuration
   webterm_server(host='localhost', port=5000, debug=True, users_db_path='custom_users.csv', log_level='DEBUG')


Using command line interface:

.. code-block:: bash

    python app.py --host localhost --port 5000 --proxy --debug --log-level DEBUG

"""

import argparse
import os
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from flask import Blueprint
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
from plant3dvision.webterm.auth import users_csv_path
from plant3dvision.webterm.terminal import terminal_manager
from plantdb.commons.fsdb import FSDB
from romitask.log import DEFAULT_LOG_LEVEL
from romitask.log import LOG_LEVELS
from romitask.log import get_logger

# Load environment variables from .env file
load_dotenv(verbose=False, override=True)

DEFAULT_ROMI_CFG = "/myapp/cfg"


def cfg_toml_path(username):
    """
    Get the configuration file directory for a given user.

    Combine the ``ROMI_CFG`` environment variable, or default value, with the provided `username`.

    Parameters
    ----------
    username : str
        The name of the user whose configuration file directory is being requested.

    Returns
    -------
    pathlib.Path
        The full path to the specified user's configuration file directory.
    """
    return Path(os.environ.get('ROMI_CFG', DEFAULT_ROMI_CFG)) / username


def get_url_plantdb():
    return os.environ.get("PLANTDB_API", "/api")

def get_url_prefix():
    return os.environ.get('WEBTERM_PREFIX', "")


def get_secret_key(logger):
    secret_key = os.environ.get('WEBTERM_SECRET_KEY', None)
    if not secret_key:
        logger.warning("No secret key found, using a random key.")
        logger.warning("Please set the `WEBTERM_SECRET_KEY` environment variable.")
        secret_key = os.urandom(24)

    return secret_key


def create_webterm_app(proxy=False, log_level=DEFAULT_LOG_LEVEL, async_mode='threading'):
    """Create and configure the WebTerm Flask application.

    Parameters
    ----------
    proxy : bool, optional
        Boolean flag indicating whether the application is behind a reverse proxy, by default False
    users_db_path : str, optional
        Path to the users database CSV file, by default 'users.csv'
    log_level : str, optional
        Logging level, by default DEFAULT_LOG_LEVEL

    Returns
    -------
    SocketIO
        Configured SocketIO application instance
    """
    webterm_prefix = get_url_prefix()

    # Get the directory where this app.py file is located
    app_dir = os.path.dirname(os.path.abspath(__file__))

    # Initialize the Flask application with explicit template and static folders
    app = Flask("WebTerm",
                template_folder=os.path.join(app_dir, 'templates'),
                static_folder=os.path.join(app_dir, 'static'),
                static_url_path=f"{webterm_prefix}/static")

    logger = get_logger("WebTerm", log_level=log_level)

    app.secret_key = get_secret_key(logger)

    # Configure proxy settings if needed
    if proxy:
        logger.info(f"Setting up Flask application with proxy support...")
        logger.info(f"Using prefix '{webterm_prefix}' for all endpoints.")
        # App is behind one proxy that sets the -For and -Host headers.
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1, x_proto=1)
        # Set secure cookies
        app.config.update(
            SESSION_COOKIE_SECURE=True,
            SESSION_COOKIE_SAMESITE='Lax'
        )

    # Create a blueprint for all routes
    bp = Blueprint('webterm',
                   __name__,
                   url_prefix=webterm_prefix,
                   template_folder=os.path.join(app_dir, 'templates'),
                   static_folder=os.path.join(app_dir, 'static'),
                   static_url_path=f"{webterm_prefix}/static")

    # Initialize Socket.IO server with configuration
    socketio_config = {
        'async_mode': async_mode,
        'cors_allowed_origins': "*",
        'ping_timeout': 5,  # ping timeout, in seconds
        'ping_interval': 2,  # ping timeout, in seconds
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

        # Get the username from session
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
        # Get the username from session
        username = session.get('username')
        if username:
            logger.info(f"User {username} disconnected")
            # Don't close terminal on disconnect - preserve state for reconnection

    @socketio.on('resize')
    def handle_resize(data):
        """Handle terminal resize events from the client."""
        # Get the username from session
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
        # Get the username from session
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
        """
        Start output polling.

        This function handles the 'start_output_polling' Socket.IO event, which is triggered when a client requests
        to start polling for terminal output.
        The function marks the polling as active for the session and initiates a background task to poll the terminal
        output at regular intervals.
        While the polling is active, the function continuously retrieves output from the terminal and emits it to
        the client via a Socket.IO event.
        The polling loop runs cooperatively using `eventlet.sleep` to avoid blocking other tasks.

        Raises
        ------
        Exception
            If there is an error retrieving or emitting the terminal output, an exception is logged and the polling loop breaks.
        """
        # Get the username from session
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
        """
        Stops the output polling for a specific user's terminal.

        This function is an event handler that listens for 'stop_output_polling' events from Socket.IO clients.
        When triggered, it stops the output polling for the terminal associated with the username stored in the session.
        """
        # Get the username from session
        username = session.get('username')
        if username and username in terminal_manager.terminals:
            terminal_manager.terminals[username]['polling_active'] = False

    @bp.route('/')
    def index():
        """
        Index of the web application (entry point).

        Returns
        -------
        flask.Response
            The response object containing a redirect to the terminal if aldready authenticated.
            Otherwise, renders the 'login.html' template.
        """
        if 'username' in session:
            return redirect(url_for('webterm.terminal'))
        return render_template('login.html',
                               WEBTERM_PREFIX=get_url_prefix())

    @bp.route('/login', methods=['POST'])
    def login():
        """
        Login endpoint to authenticate users and start a session.

        Parameters
        ----------
        request : flask.Request
            Flask request object containing form data.
        session : flask.Session
            Flask session object to store user-specific data.

        Returns
        -------
        flask.Response
            The response object containing redirect to the terminal if authentication is successful.
            Otherwise, renders the 'login.html' template with an error message.
        """
        # Retrieves the 'username' and 'password' from the request form data
        username = request.form.get('username')
        password = request.form.get('password')
        # Try to authenticate the user, returning user info dict if successful
        user = authenticate_user(username, password)
        # If successful, start a session and redirects to the terminal
        if user:
            session['username'] = username
            session['full_name'] = user['full_name']
            logger.info(f"User {username} logged in successfully")
            return redirect(url_for('webterm.terminal'))

        # Else returns an error if credentials are invalid
        logger.warning(f"Failed login attempt for username: {username}")
        return render_template('login.html',
                               error='Invalid credentials',
                               WEBTERM_PREFIX=get_url_prefix())

    @bp.route('/logout')
    def logout():
        """
        Logout Route

        This route handles user logout, clearing the session and closing any open terminals associated with the user.

        Returns
        -------
        flask.Response
            A response object that redirects to the index page after logging out.
        """
        # Get the username from session
        username = session.get('username')
        if username:
            # Close the terminal when user logs out
            terminal_manager.close_terminal(username)
            logger.info(f"User {username} logged out")

        session.clear()
        return redirect(url_for('webterm.index'))

    @bp.route('/terminal')
    def terminal():
        """
        Route for terminal page, handles requests to the '/terminal' endpoint.

        Returns
        -------
        flask.Response
            The response object containing a redirect to the index if not authenticated.
            Otherwise, renders the 'terminal.html' template.
        """
        # Pass PLANTDB_API environment variable to the template context
        # Default to local access to the PlantDB, see '/api/scans' route
        plantdb_api = get_url_plantdb()
        if 'username' not in session:
            return redirect(url_for('webterm.index'))
        return render_template('terminal.html',
                               full_name=session.get('full_name'),
                               username=session.get('username'),
                               WEBTERM_PREFIX=get_url_prefix(),
                               PLANTDB_API=plantdb_api)

    @bp.route('/api/admin')
    def admin_panel():
        """
        Route for accessing the admin panel.

        Returns
        -------
        flask.Response
            The response object containing either a redirect to the index page or the rendered admin.html template.
        """
        # Checks if there is a logged user and it is an admin
        if 'username' not in session or session.get('username') != 'admin':
            # If not, redirect to the index page
            return redirect(url_for('webterm.index'))
        # Render the admin panel template
        return render_template('admin.html',
                               WEBTERM_PREFIX=get_url_prefix())

    @bp.route('/api/admin/add_user', methods=['POST'])
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
            with open(users_csv_path(), 'a') as f:
                password_hash = hash_password(password)
                f.write(format_csv_line(full_name, username, password_hash))

            return {'success': True}, 200
        except Exception as e:
            return {'success': False, 'error': str(e)}, 500

    # Route to locally access the PlantDB database
    @bp.route('/api/scans', methods=['GET'])
    def get_scans():
        try:
            db = FSDB(os.getenv('ROMI_DB', '/myapp/db'))
            db.connect(unsafe=True)
            list_scan_names = db.list_scans(owner_only=False)
            db.disconnect()
            return jsonify(list_scan_names)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @bp.route('/api/list-toml-files', methods=['GET'])
    def list_toml_files():
        # Get the username from session
        username = session.get('username')
        # Get the directory from the environment variable or use default
        save_dir = cfg_toml_path(username)

        try:
            # Ensure directory exists
            os.makedirs(save_dir, exist_ok=True)

            # Get all TOML files in the directory
            files = [str(f) for f in os.listdir(save_dir) if f.lower().endswith('.toml')]

            return jsonify({
                'success': True,
                'files': files,
                'directory': str(save_dir)
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @bp.route('/api/load-toml-file', methods=['GET'])
    def load_toml_file():
        filename = request.args.get('filename')

        if not filename:
            return jsonify({
                'success': False,
                'error': 'Filename is required'
            }), 400

        # Get username from session
        username = session.get('username')
        # Get the directory from the environment variable or use default
        save_dir = cfg_toml_path(username)

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

    @bp.route('/api/save-toml', methods=['POST'])
    def save_toml():
        data = request.json
        filename = data.get('filename')
        content = data.get('content')

        # Get username from session
        username = session.get('username')
        # Get the directory from the environment variable or use default
        save_dir = cfg_toml_path(username)

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

    @bp.route('/api/user/profile')
    def user_profile():
        if 'username' not in session:
            return redirect(url_for('webterm.index'))
        return render_template('user_profile.html',
                               full_name=session.get('full_name'),
                               username=session.get('username'),
                               WEBTERM_PREFIX=get_url_prefix())

    @bp.route('/api/user/change_password', methods=['POST'])
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
            from plant3dvision.webterm.auth import verify_password
            if not verify_password(users[username]['password_hash'], current_password):
                return {'success': False, 'error': 'Current password is incorrect'}, 401

            # Update password in CSV
            from plant3dvision.webterm.auth import hash_password
            new_password_hash = hash_password(new_password)
            users[username]['password_hash'] = new_password_hash

            # Write all users back to CSV
            with open(users_csv_path(), 'w') as f:
                f.write('"full_name";"username";"password_hash"\n')  # Header
                for user, data in users.items():
                    f.write(format_csv_line(data['full_name'], user, data['password_hash']))

            return {'success': True}, 200
        except Exception as e:
            return {'success': False, 'error': str(e)}, 500

    # Register the blueprint with the app
    app.register_blueprint(bp)

    # Error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        """
        Handler for Not Found Error (HTTP 404) using a custom error page template (`error.html`)

        Parameters
        ----------
        error : Exception
            The exception object that was raised.
            This parameter is automatically provided by Flask when using `@app.errorhandler`.

        Returns
        -------
        flask.Response
            The response object containing the rendered error.html template.
        int
           The HTTP status code 404.
        """
        logger.error(f"Not Found Error (404): {error}")
        return render_template('error.html', error_code=404, error_message="Page not found"), 404

    @app.errorhandler(500)
    def internal_error(error):
        """
        Handler for Internal Server Error (HTTP 500) errors using a custom error page template (`error.html`)
        Parameters
        ----------
        error : Exception
            The exception object that was raised.
            This parameter is automatically provided by Flask when using `@app.errorhandler`.

        Returns
        -------
        flask.Response
            The response object containing the rendered error.html template.
        int
           The HTTP status code 500.
        """
        logger.error(f"Internal Server Error (500): {error}")
        return render_template('error.html', error_code=500, error_message="Internal server error"), 500

    # Create users.csv if it doesn't exist
    if not os.path.exists(users_csv_path()):
        logger.warning("No existing users database found, creating a new one.")
        with open(users_csv_path(), 'w') as f:
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
    server_args.add_argument('--host', type=str,
                             default=os.environ.get('SERVER_HOST', '0.0.0.0'),
                             help="the hostname to listen on, defaults to '0.0.0.0'.")
    server_args.add_argument('--port', type=int,
                             default=int(os.environ.get('SERVER_PORT', 8080)),
                             help="the port of the webserver, defaults to '8080'.")
    server_args.add_argument('--proxy', action='store_true',
                             help="use this flag when this server sits behind a reverse proxy")
    server_args.add_argument('--url-prefix', type=str,
                             default=os.environ.get('WEBTERM_PREFIX', ''),
                             help="prefix for the webserver URL, defaults to ''.")
    server_args.add_argument('--debug', action='store_true',
                             help="enable debug mode.")

    auth_args = parser.add_argument_group("authentication arguments")
    auth_args.add_argument('--users-db', dest='users_db_path', type=str,
                           default=os.environ.get('WEBTERM_USERS', 'users.csv'),
                           help="path to the users database CSV file, defaults to 'users.csv'.")
    auth_args.add_argument('--secret-key', dest='secret_key', type=str,
                           default=os.environ.get('WEBTERM_SECRET_KEY'),
                           help="secret key for session management.")

    log_opt = parser.add_argument_group("logging options")
    log_opt.add_argument("--log-level", dest="log_level", type=str,
                         default=DEFAULT_LOG_LEVEL, choices=LOG_LEVELS,
                         help="level of message logging, defaults to 'INFO'.")

    return parser


def webterm_server(host='0.0.0.0', port=8080, proxy=False, debug=False, log_level=DEFAULT_LOG_LEVEL):
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
    log_level : str, optional
        Logging level, by default DEFAULT_LOG_LEVEL
    """

    # Create the application using the factory function
    socketio_app, flask_app = create_webterm_app(
        proxy=proxy,
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

    if args.webterm_prefix != os.environ.get('WEBTERM_PREFIX', ''):
        os.environ['WEBTERM_PREFIX'] = args.webterm_prefix
    if args.users_db != os.environ.get('WEBTERM_USERS', ''):
        os.environ['WEBTERM_USERS'] = args.users_db
    if args.secret_key != os.environ.get('WEBTERM_SECRET_KEY', ''):
        os.environ['WEBTERM_SECRET_KEY'] = args.secret_key

    webterm_server(
        host=args.host,
        port=args.port,
        proxy=args.proxy,
        debug=args.debug,
        log_level=args.log_level
    )


if __name__ == '__main__':
    main()
