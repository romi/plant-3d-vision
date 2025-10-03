#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Authentication Module for User Management System

A secure authentication module that provides user credential management and verification functionality using bcrypt hashing.
This module handles user data storage in CSV format and offers robust password hashing and verification mechanisms.

## Key Features

- Secure password hashing using bcrypt
- User data management with CSV file storage
- User authentication against stored credentials
- Safe password verification
- CSV formatting utilities for user data

## Usage Examples

```python
>>> # Hash a new password
>>> hashed_password = hash_password("mypassword123")

>>> # Authenticate a user
>>> user_info = authenticate_user("johndoe", "mypassword123")
>>> if user_info:
...     print(f"Welcome, {user_info['full_name']}!")
... else:
...     print("Authentication failed")

>>> # Add a new user to CSV
>>> csv_line = format_csv_line("John Doe", "johndoe", hashed_password)
>>> with open('users.csv', 'a') as f:
...     f.write(csv_line)
```
"""

import csv
import os
from pathlib import Path

import bcrypt

DEFAULT_ROMI_USERS = "/myapp/users"


def users_csv_path():
    """
    Return the path to the CSV file that contains user data.

    The function checks for an environment variable `ROMI_USERS` and uses its value as
    the base directory path. If the environment variable is not set, it defaults to a predefined
    path. The resulting path points to a CSV file named 'users.csv' within the specified or default
    directory.

    Returns
    -------
    pathlib.Path
        The absolute path to the users CSV file.
    """
    return Path(os.environ.get("ROMI_USERS", DEFAULT_ROMI_USERS)) / 'users.csv'


def hash_password(password):
    """
    Hash the given password using bcrypt.

    Parameters
    ----------
    password : str
        The password to be hashed.
        It should be a string containing the user's password.

    Return
    -------
    str
        A string representing the hashed password.
    """
    # Convert password to bytes if it's not already
    if isinstance(password, str):
        password = password.encode('utf-8')

    # Generate a salt and hash the password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password, salt)

    # Return the hash as a string
    return hashed.decode('utf-8')


def load_users():
    """
    Loads user data from a CSV file into a dictionary.

    Parameters
    ----------
    None

    Returns
    -------
    dict
        A dictionary where the keys are usernames and the values are dictionaries with full_name and password_hash for each user.
        If the file does not exist or cannot be read, it returns an empty dict.
    """
    users = {}
    if not os.path.exists(users_csv_path()):
        return users

    with open(users_csv_path(), 'r') as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 3:
                full_name, username, password_hash = row[0], row[1], row[2]
                users[username] = {
                    'full_name': full_name,
                    'password_hash': password_hash
                }
    return users


def authenticate_user(username, password):
    """
    Authenticate a user by checking their credentials against stored data.

    Parameters
    ----------
    username : str
       The username to authenticate. Must be a valid username in the system.
    password : str
       The password to authenticate. Should match the stored hash for the given username.

    Returns
    -------
    dict or None
        If authentication is successful, returns a dictionary with 'username' and 'full_name'.
        Otherwise, returns None.
    """
    users = load_users()
    if username in users:
        if verify_password(users[username]['password_hash'], password):
            return {
                'username': username,
                'full_name': users[username]['full_name']
            }
    return None


def verify_password(stored_hash, password):
    """
    Verify a password against a stored hash using bcrypt.

    Parameters
    ----------
    stored_hash : str
        The stored password hash to verify against.
    password : str
        The plain text password to verify.

    Returns
    -------
    bool
        True if the password matches the hash, False otherwise.
    """
    # Convert inputs to bytes if they're not already
    if isinstance(password, str):
        password = password.encode('utf-8')
    if isinstance(stored_hash, str):
        stored_hash = stored_hash.encode('utf-8')

    try:
        # Use bcrypt's checkpw function to verify the password
        return bcrypt.checkpw(password, stored_hash)
    except Exception:
        # Return False if there's any error (e.g., invalid hash format)
        return False


def format_csv_line(full_name, username, password_hash):
    """
    Format a user's data into a CSV line.

    Parameters
    ----------
    full_name : str
        The user's full name.
    username : str
        The user's username.
    password_hash : str
        The hashed password of the user.

    Returns
    -------
    str
        A string representing the CSV line formatted as described above.

    Examples
    --------
    >>> format_csv_line('John Doe', 'johndoe', 'hashed_password')
    '"John Doe";"johndoe";"hashed_password"\n'
    """
    return f'"{full_name}";"{username}";"{password_hash}"\n'
