#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

    plant3dvision - Python tools for the ROMI 3D Scanner

    Copyright (C) 2018 Sony Computer Science Laboratories
    Authors: D. Colliaux, T. Wintz, P. Hanappe
  
    This file is part of plant3dvision.

    plant3dvision is free software: you can redistribute it
    and/or modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation, either
    version 3 of the License, or (at your option) any later version.

    plant3dvision is distributed in the hope that it will be
    useful, but WITHOUT ANY WARRANTY; without even the implied
    warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
    See the GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with plant3dvision.  If not, see
    <https://www.gnu.org/licenses/>.

"""
import logging
from os import path

from colorlog import ColoredFormatter


def configure_logger(name, log_path="", log_level='INFO'):
    colored_formatter = ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(bg_blue)s[%(name)s]%(reset)s %(message)s",
        datefmt=None,
        reset=True,
        style='%'
    )
    simple_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    # create console handler:
    console = logging.StreamHandler()
    console.setFormatter(colored_formatter)

    logger = logging.getLogger(name)
    logger.addHandler(console)
    logger.setLevel(getattr(logging, log_level))

    if log_path is not None and log_path != "":
        # create file handler:
        fh = logging.FileHandler(path.join(log_path, f'{name}.log'), mode='w')
        fh.setFormatter(simple_formatter)
        logger.addHandler(fh)

    return logger


logger = configure_logger('plant3dvision')

