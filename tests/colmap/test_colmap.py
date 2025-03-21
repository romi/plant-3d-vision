#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path
from tempfile import mkdtemp

import docker

from plantdb.commons.test_database import setup_test_database

TMP_TEST_DIR = mkdtemp(prefix="test_colmap_")


def parsing():
    parser = argparse.ArgumentParser(description='Test a Colmap image..')
    parser.add_argument('colmap_image', type=str, default='roboticsmicrofarms/colmap',
                        help='name of the image to use')
    parser.add_argument('-c', '--colmap', dest='colmap_version', type=str, default='3.8',
                        help='colmap version')
    parser.add_argument('-t', '--tag', type=str, default='3.8',
                        help='image tag')
    return parser


# colmap_image = "roboticsmicrofarms/colmap"
# tag = "31df46c6"  # roboticsmicrofarms/colmap:31df46c6
# colmap_image = "colmap/colmap"
# tag = "3.8"  # colmap/colmap:3.8

# colmap_version = "3.8"


def main(args):
    colmap_image = args.colmap_image
    colmap_version = args.colmap_version
    tag = args.tag

    # Set up a test database with the 'real_plant' dataset (pulled from ZENODO):
    db_path = setup_test_database(
        ['real_plant'],
        out_path=TMP_TEST_DIR
    )
    log_file = db_path / f"colmap.log"

    # -----------------------------------------------------------------------------
    # EXTRACT POSES:
    # -----------------------------------------------------------------------------

    # Extract the poses from the images metadata:
    posefile = open(f"/{db_path}/real_plant/poses.txt", mode='w')
    # - Try to get the pose from each file metadata:
    for i, file in enumerate(sorted(os.listdir(f"/{db_path}/real_plant/metadata/images"))):
        with open(f"/{db_path}/real_plant/metadata/images/{file}", mode='r') as f:
            jdict = json.load(f)
        # print(jdict)
        try:
            p = jdict['approximate_pose']
        except KeyError:
            p = jdict['pose']  # backward compatibility, should work for provided test dataset
        s = '%s %d %d %d\n' % (file.split('.')[0] + ".jpg", p[0], p[1], p[2])
        posefile.write(s)

    posefile.close()

    # -----------------------------------------------------------------------------
    # PARSE COLMAP COMMANDS
    # -----------------------------------------------------------------------------

    # Parse files with colmap bash commands:
    current_path = Path(__file__).parent
    with open(current_path / f'test_colmap{colmap_version}.sh', 'r') as cmd_f:
        colmap_cmd = "".join(cmd_f.readlines())

    colmap_cmd_list = colmap_cmd.split('\n\n')
    if colmap_cmd_list[0].startswith('#!'):
        colmap_cmd_list = colmap_cmd_list[1:]

    # -----------------------------------------------------------------------------
    # DOCKER
    # -----------------------------------------------------------------------------

    # Initialize docker client manager:
    client = docker.from_env()

    # Defines environment variables:
    varenv = {}
    varenv.update({
        'PYOPENCL_CTX': os.environ.get('PYOPENCL_CTX', '0'),
    })

    # Defines the mount point:
    mount = docker.types.Mount(str(db_path), str(db_path), type='bind')

    # Get the GPU device:
    gpu_device = docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])

    # Call colmap commands in docker container:
    for cmd in colmap_cmd_list:
        cmd = cmd.replace('$DATASET_PATH', str(db_path / 'real_plant'))
        print(cmd)
        out = client.containers.run(colmap_image + f":{tag}", cmd, environment=varenv, mounts=[mount],
                                    stdout=True, stderr=True, device_requests=[gpu_device])
        with open(log_file, mode="a") as f:
            f.writelines(out.decode('utf8'))


if __name__ == '__main__':
    parser = parsing()
    args = parser.parse_args()
    main(args)
