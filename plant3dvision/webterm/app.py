# !/usr/bin/env python
# -*- coding: utf-8 -*-


import os

from dotenv import load_dotenv
from flask import Flask
from flask import jsonify
from flask import redirect
from flask import render_template
from flask import request
from flask import session
from flask import url_for
from flask_socketio import SocketIO
from plantdb.commons.fsdb import FSDB

from auth import authenticate_user
from auth import format_csv_line
from auth import hash_password
from auth import load_users
from terminal import read_terminal_output
from terminal import create_terminal
from terminal import handle_terminal_input

# Load environment variables from .env file
load_dotenv(verbose=False, override=True)

# Initialize Flask application
app = Flask(__name__)
app.secret_key = os.environ.get('SERVER_SECRET_KEY', os.urandom(24))
socketio = SocketIO(app, async_mode='eventlet')

# Store active terminals
terminals = {}


# Socket.IO event handlers
@socketio.on('connect')
def handle_connect():
    if 'username' not in session:
        return False

    username = session.get('username')
    # Create a new terminal for this user if one doesn't exist
    if username not in terminals:
        terminals[username] = create_terminal()
        # Change to $ROMI_DB directory automatically
        handle_terminal_input(terminals[username], {'input': 'cd $ROMI_DB\n'})

    return True


@socketio.on('disconnect')
def handle_disconnect():
    username = session.get('username', None)
    if username and username in terminals:
        # Don't actually close the terminal on disconnect to preserve state
        # between page refreshes, only on logout
        pass


@socketio.on('terminal_input')
def socket_handle_terminal_input(data):
    username = session.get('username')
    if not username or username not in terminals:
        return

    terminal = terminals[username]
    output = handle_terminal_input(terminal, data)
    socketio.emit('terminal_output', {'output': output}, room=request.sid)


@socketio.on('start_output_polling')
def start_output_polling():
    username = session.get('username')
    if not username or username not in terminals:
        return

    # Get the current request.sid and store it
    sid = request.sid

    terminal = terminals[username]

    # Store a flag in terminal dict to track if polling should continue
    terminal['polling_active'] = True
    # Store the session ID
    terminal['sid'] = sid

    def poll_output():
        while terminal.get('polling_active', False):
            try:
                # Check if there's any output available
                output = read_terminal_output(terminal['main'])
                if output:
                    # Use the stored sid instead of request.sid
                    socketio.emit('terminal_output', {'output': output}, room=terminal['sid'])
            except Exception as e:
                # Use the stored sid instead of request.sid
                socketio.emit('terminal_output',
                              {'output': f"\r\nError in polling: {str(e)}\r\n"},
                              room=terminal['sid'])
                break
            # Sleep a short time to avoid consuming too much CPU
            socketio.sleep(0.1)

    # Start polling in a background task
    socketio.start_background_task(poll_output)


@socketio.on('stop_output_polling')
def stop_output_polling():
    username = session.get('username')
    if username and username in terminals:
        terminals[username]['polling_active'] = False


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
        return redirect(url_for('terminal'))
    return render_template('login.html', error='Invalid credentials')


@app.route('/logout')
def logout():
    # Close terminal if exists
    if session.get('username') in terminals:
        # Close the terminal process
        terminals.pop(session.get('username'), None)

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
        with open('users.csv', 'a') as f:
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
        with open('users.csv', 'w') as f:
            f.write('"full_name";"username";"password_hash"\n')  # Header
            for user, data in users.items():
                f.write(format_csv_line(data['full_name'], user, data['password_hash']))

        return {'success': True}, 200
    except Exception as e:
        return {'success': False, 'error': str(e)}, 500


if __name__ == '__main__':
    print('Starting WebTerm server...')
    # Create users.csv if it doesn't exist
    if not os.path.exists('users.csv'):
        with open('users.csv', 'w') as f:
            f.write(format_csv_line("full_name", "username", "password_hash"))
            # Add default admin user
            admin_hash = hash_password('admin')
            f.write(format_csv_line("Administrator", "admin", admin_hash))

    # Start the server
    host = os.environ.get('SERVER_HOST', '0.0.0.0')
    port = int(os.environ.get('SERVER_PORT', 8080))
    print(f"Starting server on http://{host}:{port}")
    socketio.run(app, host=host, port=port)
    print('WebTerm server stopped!')