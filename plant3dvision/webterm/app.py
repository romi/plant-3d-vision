#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pty
import threading
from pathlib import Path

import docker
import requests
from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request
from flask import session
from flask_cors import CORS
from flask_socketio import SocketIO
from flask_socketio import emit
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent')

# Initialize Docker client
docker_client = docker.from_env()

# App configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'zip'}  # Only allow zip files
REST_API_BASE_URL = os.environ.get('REST_API_BASE_URL', 'http://localhost:5000')
TERMINAL_CONTAINER = os.environ.get('TERMINAL_CONTAINER', 'plant3dvision')
CONTAINER_USER = os.environ.get('CONTAINER_USER', 'romi')

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class DockerPTY:
    def __init__(self, docker_client, container_name, user=None):
        self.docker_client = docker_client
        self.container = docker_client.containers.get(container_name)
        self.pty_master, self.pty_slave = pty.openpty()
        self.exec_id = None
        self.socket = None
        self.user = user

    def start(self):
        cmd = '/bin/bash'
        if self.user:
            cmd = f'su - {self.user}'

        # Create exec instance
        exec_create = self.container.client.api.exec_create(
            self.container.id,
            cmd,
            stdin=True,
            stdout=True,
            stderr=True,
            tty=True,
            privileged=True
        )

        # Start exec instance and get socket
        self.exec_id = exec_create['Id']
        self.socket = self.container.client.api.exec_start(
            self.exec_id,
            socket=True,
            tty=True,
            demux=False
        )._sock

        return self.pty_master, self.pty_slave

    def resize(self, rows, cols):
        if self.exec_id:
            self.container.client.api.exec_resize(
                self.exec_id,
                height=rows,
                width=cols
            )


def allowed_file(filename):
    # Check if file has an extension and it's in allowed extensions
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    # Render index page with authentication status
    if 'user_id' not in session:
        return render_template('index.html', authenticated=False)
    return render_template('index.html', authenticated=True)


@app.route('/login', methods=['POST'])
def login():
    # Handle user login via REST API
    data = request.json
    username = data.get('username')
    password = data.get('password')

    response = requests.post(f'{REST_API_BASE_URL}/login',
                             json={'username': username, 'password': password})
    if response.status_code == 200:
        session['user_id'] = username
        return jsonify({'success': True})
    return jsonify({'success': False, 'message': 'Invalid credentials'}), 401


@app.route('/upload', methods=['POST'])
def upload_file():
    # Handle file upload and forward to REST API
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        scan_id = Path(filename).stem  # Use filename without extension as scan ID

        # Save file temporarily
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            # Forward file to REST API
            with open(filepath, 'rb') as f:
                response = requests.post(
                    f'{REST_API_BASE_URL}/archive/{scan_id}',
                    files={'file': f}
                )
            # Clean up temporary file
            os.remove(filepath)

            if response.status_code == 200:
                return jsonify({'success': True, 'scan_id': scan_id})
            else:
                return jsonify({
                    'error': f'REST API error: {response.status_code}',
                    'message': response.text
                }), response.status_code
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/favicon.ico')
def favicon():
    return """
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-terminal" viewBox="0 0 16 16">
      <path d="M6 9a.5.5 0 0 1 .5-.5h3a.5.5 0 0 1 0 1h-3A.5.5 0 0 1 6 9M3.854 4.146a.5.5 0 1 0-.708.708L4.793 6.5 3.146 8.146a.5.5 0 1 0 .708.708l2-2a.5.5 0 0 0 0-.708z"/>
      <path d="M2 1a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V3a2 2 0 0 0-2-2zm12 1a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H2a1 1 0 0 1-1-1V3a1 1 0 0 1 1-1z"/>
    </svg>
    """, {'Content-Type': 'image/svg+xml'}


# Store terminals in a dictionary with socket ID as key
terminals = {}


@socketio.on('connect')
def handle_connect():
    try:
        terminal = DockerPTY(docker_client, TERMINAL_CONTAINER, user=CONTAINER_USER)
        terminal.start()

        # Store the socket ID at connection time
        sid = request.sid
        terminals[sid] = terminal

        def read_and_forward_pty_output(socket_id):
            max_read_bytes = 1024 * 20
            while True:
                try:
                    if socket_id not in terminals:
                        break
                    terminal = terminals[socket_id]
                    if terminal and terminal.socket:
                        output = terminal.socket.recv(max_read_bytes)
                        if output:
                            # Use the stored socket ID instead of request.sid
                            with app.app_context():
                                socketio.emit('terminal_output',
                                              {'output': output.decode()},
                                              room=socket_id)
                        else:
                            break
                except Exception as e:
                    print(f"Error reading from terminal: {e}")
                    with app.app_context():
                        socketio.emit('terminal_output',
                                      {'error': str(e)},
                                      room=socket_id)
                    break

        # Pass the socket ID to the thread
        thread = threading.Thread(target=read_and_forward_pty_output, args=(sid,))
        thread.daemon = True
        thread.start()

    except Exception as e:
        emit('terminal_output', {'error': str(e)})


@socketio.on('disconnect')
def handle_disconnect():
    if request.sid in terminals:
        terminal = terminals[request.sid]
        if terminal and terminal.socket:
            terminal.socket.close()
        del terminals[request.sid]


@socketio.on('terminal_input')
def handle_terminal_input(data):
    terminal = terminals.get(request.sid)
    if not terminal or not terminal.socket:
        return

    try:
        input_data = data.get('input', '')
        terminal.socket.send(input_data.encode())
    except Exception as e:
        emit('terminal_output', {'error': str(e)})


@socketio.on('terminal_resize')
def handle_terminal_resize(data):
    terminal = terminals.get(request.sid)
    if terminal:
        try:
            rows = data.get('rows', 24)
            cols = data.get('cols', 80)
            terminal.resize(rows, cols)
        except Exception as e:
            emit('terminal_output', {'error': str(e)})


@app.route('/logout', methods=['POST'])
def logout():
    # Clear session data
    session.clear()
    return jsonify({'success': True})


if __name__ == '__main__':
    socketio.run(
        app,
        debug=True,  # enable debug mode
        host='0.0.0.0',  # to listen to all network interfaces
        port=5001  # to avoid conflict with PlantDB REST API (5000)
    )
