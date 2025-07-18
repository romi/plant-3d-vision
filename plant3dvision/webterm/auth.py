#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#  Copyright (c) 2022 Univ. Lyon, ENS de Lyon, UCB Lyon 1, CNRS, INRAe, Inria
#  All rights reserved.
#  This file is part of the TimageTK library, and is released under the "GPLv3"
#  license. Please see the LICENSE.md file that should have been included as
#  part of this package.
# ------------------------------------------------------------------------------

import csv
import os
from passlib.hash import bcrypt_sha256


def hash_password(password):
    """Hash a password using bcrypt."""
    # h_name, h_version, h_type, h_round, h_salt, h_digest = bcrypt_sha256.hash(password).split("$")
    return bcrypt_sha256.hash(password)


def load_users():
    """Load users from CSV file."""
    users = {}
    if not os.path.exists('users.csv'):
        return users

    with open('users.csv', 'r') as f:
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
    """Authenticate a user with username and password."""
    users = load_users()
    if username in users:
        if bcrypt_sha256.verify(password, users[username]['password_hash']):
            return {
                'username': username,
                'full_name': users[username]['full_name']
            }
    return None


def verify_password(stored_hash, password):
    """Verify a password against a stored hash."""
    return bcrypt_sha256.verify(password, stored_hash)

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