# !/usr/bin/env python
# -*- coding: utf-8 -*-


import os

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
from auth import hash_password
from plant3dvision.webterm.auth import format_csv_line
from plant3dvision.webterm.auth import load_users
from terminal import create_terminal
from terminal import handle_terminal_input

# Initialize Flask application
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))
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
        # db = FSDB(os.getenv('ROMI_DB', '/myapp/db'))
        db = FSDB(os.getenv('ROMI_DB', '/data/ROMI/test_owner'))
        db.connect(unsafe=True)
        list_scan_names = db.list_scans(owner_only=False)
        db.disconnect()
        return jsonify(list_scan_names)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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

        # Verify current password
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
    # Create users.csv if it doesn't exist
    if not os.path.exists('users.csv'):
        with open('users.csv', 'w') as f:
            f.write(format_csv_line("full_name", "username", "password_hash"))
            # Add default admin user
            admin_hash = hash_password('admin')
            f.write(format_csv_line("Administrator", "admin", admin_hash))

    # Start the server
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 8080))
    socketio.run(app, host=host, port=port)
