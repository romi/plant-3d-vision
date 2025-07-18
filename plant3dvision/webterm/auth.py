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
    """
    Hash the given password using bcrypt and SHA-256.

    Parameters
    ----------
    password : str
        The password to be hashed.
        It should be a string containing the user's password.

    Return
    -------
    str
        A string representing the hashed password, including the algorithm identifier, rounds count,
        salt, and digest separated by dollar signs.
    """
    # h_name, h_version, h_type, h_round, h_salt, h_digest = bcrypt_sha256.hash(password).split("$")
    return bcrypt_sha256.hash(password)


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
        If the file does not exist or cannot be read, returns an empty dict.
    """
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
        if bcrypt_sha256.verify(password, users[username]['password_hash']):
            return {
                'username': username,
                'full_name': users[username]['full_name']
            }
    return None


def verify_password(stored_hash, password):
    """
    Compute a hash from a string using bcrypt_sha256 algorithm.

    Parameters
    ----------
    password : str
        The input string to be hashed.
    salt : bytes, optional
        Optional salt value to use in addition to the built-in salt.
        If None, a new random salt will be generated and used.

    Returns
    -------
    bytes
        The resulting hash as raw bytes

    See Also
    --------
    passlib.hash.bcrypt_sha256
        The underlying cryptographic algorithm used for hashing.
    """
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