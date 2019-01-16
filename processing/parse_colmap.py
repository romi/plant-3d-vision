import numpy as np

def parse_cameras_file(cameras_str):
    camera = {}
    for line in cameras_str.splitlines():
        if line[0] == '#':
            continue
        sp = line.split()
        if sp[0] == '1':
            camera['width'] = int(sp[2])
            camera['height'] = int(sp[3])
            if sp[1] == 'SIMPLE_RADIAL':
                camera['parameters'] = {'fx' : float(sp[4]),
                                        'fy' : float(sp[4]),
                                        'cx' : float(sp[5]),
                                        'cy': float(sp[6]),
                                        'k1' : float(sp[7]),
                                        'k2' : float(sp[7]),
                                        'p1' : 0.,
                                        'p2' : 0.}
            elif sp[1] == 'RADIAL':
                camera['parameters'] = {'fx' : float(sp[4]),
                                        'fy' : float(sp[4]),
                                        'cx' : float(sp[5]),
                                        'cy': float(sp[6]),
                                        'k1' : float(sp[7]),
                                        'k2' : float(sp[8]),
                                        'p1' : 0.,
                                        'p2' : 0.}
            elif sp[1] == 'OPENCV':
                 camera['parameters'] = {'fx' : float(sp[4]),
                                         'fy' : float(sp[5]),
                                        'cx' : float(sp[6]),
                                        'cy': float(sp[7]),
                                        'k1' : float(sp[8]),
                                        'k2' : float(sp[9]),
                                        'p1' : float(sp[10]),
                                        'p2' : float(sp[11])}
            else:
                raise Exception('Unsupported camera type')
            break
        line = file_camera.readline()
        cnt += 1
    return camera

def parse_images_file(images_str):
    images = []
    for cnt, line in enumerate(images_str.splitlines()):
        sp = line.split()
        if cnt % 2 == 0 and cnt >= 4:
            pose = {}
            key = sp[0]
            fname = sp[-1]
            qw, qx, qy, qz, tx, ty, tz = [float(x) for x in sp[1:8]]
            pose['rotation'] = [[1 - 2*qy*qy - 2*qz*qz, 2*qx*qy-2*qz*qw, 2*qx*qz + 2*qy*qw],
                   [2*qx*qy + 2*qz*qw,1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
                   [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]]
            pose['translation'] = [tx, ty, tz]
            pose['image_key'] = key
            images.append({
                'filename' : fname,
                'pose' : pose
            })
    return images

def parse_points_file(points_str):
    images = {}
    points = np.zeros((0, 3))
    for line in points_str.splitlines():
        if line[0] == '#':
            continue
        sp = line.split()
        x, y, z = sp[1:4]
        points = np.vstack([points, [float(x), float(y), float(z)]])
    return points
