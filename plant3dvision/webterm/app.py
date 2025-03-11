#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
import docker
import os
from werkzeug.utils import secure_filename
import requests
from pathlib import Path
import json


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key
socketio = SocketIO(app)

# Docker client
docker_client = docker.from_env()

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'zip'}
REST_API_BASE_URL = os.environ.get('REST_API_BASE_URL', 'http://localhost:5000')  # Adjust to your REST API URL

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    if 'user_id' not in session:
        return render_template('index.html', authenticated=False)
    return render_template('index.html', authenticated=True)


@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    # Use existing Login endpoint
    response = requests.post(f'{REST_API_BASE_URL}/login',
                             json={'username': username, 'password': password})

    if response.status_code == 200:
        session['user_id'] = username
        return jsonify({'success': True})
    return jsonify({'success': False, 'message': 'Invalid credentials'}), 401


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        scan_id = Path(filename).stem  # Get filename without extension

        # Save file temporarily
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
            # Forward to REST API
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


@socketio.on('connect')
def handle_connect():
    if 'user_id' not in session:
        return False
    return True


@socketio.on('terminal_input')
def handle_terminal_input(data):
    if 'user_id' not in session:
        return

    try:
        container = docker_client.containers.get('plant3dvision')
        command = data.get('command', '')

        # Execute command in container
        exec_result = container.exec_run(
            cmd=f"/bin/sh -c '{command}'",
            stream=True
        )

        # Stream output back to client
        for output in exec_result.output:
            emit('terminal_output', {'output': output.decode()})

    except docker.errors.NotFound:
        emit('terminal_output', {'error': 'Container not found'})
    except Exception as e:
        emit('terminal_output', {'error': str(e)})


@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})


if __name__ == '__main__':
    socketio.run(app, debug=True, port=5001)  # Using different port than REST API
