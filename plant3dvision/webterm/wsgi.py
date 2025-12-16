#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""WSGI Application Entry Point for WebTerm

This module serves as the Web Server Gateway Interface (WSGI) entry point for the WebTerm application,
allowing web servers like Gunicorn, uWSGI, or Apache with mod_wsgi to interact with the application.

Environment Variables
---------------------

App Configuration Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``WEBTERM_SECRET_KEY``: Secret key for Flask sessions (default: randomly generated)
- ``WEBTERM_USERS``: Path to users CSV file (default: ``/myapp/users``)
- ``ROMI_DB``: Path to ROMI database (default: ``/myapp/db``)
- ``ROMI_CFG``: Path to configuration directory (default: ``/myapp/cfg/``)
- ``WEBTERM_PROXY``: Set to 'true' if behind a reverse proxy, (default ``False``)
- ``WEBTERM_PREFIX``: Prefix for WebTerm routes (default: ``''``)
- ``LOG_LEVEL``: Logging level (default: ``INFO``)

Additional Environment Variables for development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``SERVER_HOST``: Host to bind to (default: ``0.0.0.0``)
- ``SERVER_PORT``: Port to bind to (default: ``8080``)
- ``SERVER_DEBUG``: Enable debug mode (default: ``False``)

Usage with Gunicorn
-------------------
After installing Gunicorn, you can run the application using the following command:

.. code-block:: bash

   gunicorn --worker-class eventlet -w 1 --bind ${SERVER_HOST:-0.0.0.0}:${SERVER_PORT:-8080} plant3dvision.webterm.wsgi:application

This will start the application on port ``SERVER_HOST:SERVER_PORT`` or ``0.0.0.0:8080`` by default.
"""

import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv(verbose=False, override=True)

# Disable eventlet multiple readers check for terminal operations
try:
    from eventlet.debug import hub_prevent_multiple_readers

    hub_prevent_multiple_readers(False)
except ImportError:
    pass

# Import the application factory function
from plant3dvision.webterm.app import create_webterm_app
from romitask.log import DEFAULT_LOG_LEVEL

# Get configuration from environment variables
app_config = {
    'proxy': os.environ.get('WEBTERM_PROXY', 'false').lower() == 'true',
    'log_level': os.environ.get('LOG_LEVEL', DEFAULT_LOG_LEVEL),
    'async_mode': 'eventlet'
}

# Create the application instances for WSGI
socketio_app, flask_app = create_webterm_app(**app_config)
# WSGI application instance (what uWSGI will use)
application = flask_app

if __name__ == '__main__':
    socketio_app, flask_app = create_webterm_app(**app_config)
    run_config = {
        'host': os.environ.get('SERVER_HOST', '0.0.0.0'),
        'port': int(os.environ.get('SERVER_PORT', 8080)),
        'debug': bool(os.environ.get('SERVER_DEBUG', False)),
    }
    # This allows running the WSGI file directly for testing
    print("Starting WebTerm in WSGI test mode...")
    socketio_app.run(flask_app, **run_config)
