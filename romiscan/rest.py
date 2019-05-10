from flask import Flask, send_file
from flask import request, send_from_directory
from flask_cors import CORS
import json
from flask_restful import Resource, Api
from lettucethink.db.fsdb import DB
import os
import io

app = Flask(__name__)
# CORS(app)
api = Api(app)

# db_location =  '/home/twintz/Data/scanner/processed'
db_location =  '/data/v0.4'
# db = DB('/home/twintz/dataviz/processed')
db = DB(db_location)
db.connect()

db_prefix = "/files"


def fmt_date(scan):
    try:
        x = scan.id
        date, time = x.split('_')
        time = time.replace('-', ':')
    except:
        date, time = "01-01-00","00:00:00"
    return "%s %s" % (date, time)


def compute_fileset_matches(scan):
    filesets_matches = {}
    for fs in scan.get_filesets():
        x = fs.id.split('_')[0]
        filesets_matches[x] = fs.id
    return filesets_matches


def fmt_scan_minimal(scan, filesets_matches):
    metadata = scan.get_metadata()
    try:
        species = metadata['object']['species']
    except:
        species = "N/A"
    try:
        environment = metadata['object']['environment']
    except:
        environment = "N/A"
    try:
        plant = metadata['object']['plant_id']
    except:
        plant = "N/A"

    n_photos = len(scan.get_fileset('images').get_files())

    fileset_visu = scan.get_fileset(filesets_matches['Visualization'])

    first_image = scan.get_fileset('images').get_files()[0].id
    thumbnail = os.path.join(
        db_prefix, scan.id, "%s/thumbnail_%s.jpg" % (filesets_matches['Visualization'], first_image))

    has_mesh = fileset_visu.get_file('mesh') is not None
    has_point_cloud = fileset_visu.get_file('pointcloud') is not None
    has_skeleton = fileset_visu.get_file('skeleton') is not None
    has_angle = fileset_visu.get_file('angles') is not None

    return {
        "id": scan.id,
        "metadata": {
            "date": fmt_date(scan),
            "species": species,
            "plant": plant,
            "environment": environment,
            "nbPhotos": n_photos,
            "files": {
                "metadatas": os.path.join(db_prefix, scan.id, "metadata/metadata.json"),
                "archive": os.path.join(db_prefix, scan.id, "%s/scan.zip" % filesets_matches['Visualization'])
            }
        },
        "thumbnailUri": thumbnail,
        "hasMesh": has_mesh,
        "hasPointCloud": has_point_cloud,
        "hasSkeleton": has_skeleton,
        "hasAngleData": has_angle
    }


def fmt_scans(scans, query):
    res = []
    for scan in scans:
        filesets_matches = compute_fileset_matches(scan)
        if 'Visualization' in filesets_matches:
            metadata = scan.get_metadata()
            if query is not None and not (query.lower() in json.dumps(metadata).lower()):
                continue
            res.append(fmt_scan_minimal(scan, filesets_matches))
    return res


def fmt_scan(scan, filesets_matches):
    res = fmt_scan_minimal(scan, filesets_matches)
    metadata = scan.get_metadata()

    files_uri = {}
    if(res["hasMesh"]):
        files_uri["mesh"] = os.path.join(
            db_prefix, scan.id, "%s/mesh.ply" % filesets_matches['Visualization'])
    if(res["hasPointCloud"]):
        files_uri["pointCloud"] = os.path.join(
            db_prefix, scan.id, "%s/pointcloud.ply" % filesets_matches['Visualization'])

    res["filesUri"] = files_uri
    res["data"] = {}

    if(res["hasSkeleton"]):
        x = json.loads(scan.get_fileset(
            filesets_matches['Visualization']).get_file('skeleton').read_text())
        res["data"]["skeleton"] = x

    if(res["hasAngleData"]):
        x = json.loads(scan.get_fileset(
            filesets_matches['Visualization']).get_file('angles').read_text())
        res["data"]["angles"] = x

    res["workspace"] = metadata["scanner"]["workspace"]

    res["camera"] = {}

    # if 'camera_model' in metadata['scanner']:
    #     res["camera"]["model"] = metadata["scanner"]["camera_model"]
    #     res["camera"]["poses"] = []
    # else:
    res["camera"]["model"] = metadata["computed"]["camera_model"]
    res["camera"]["poses"] = []

    poses = json.loads(scan.get_fileset(
        filesets_matches['Colmap']).get_file('images').read_text())

    for f in scan.get_fileset('images').get_files():
        id = f.id
        for k in poses.keys():
            if poses[k]['name'] == f.filename:
                res['camera']['poses'].append({
                    'id': f.id,
                    'tvec': poses[k]['tvec'],
                    'rotmat': poses[k]['rotmat'],
                    'photoUri': os.path.join(db_prefix, scan.id, "%s/lowres_%s.jpg" % (filesets_matches['Visualization'] ,f.id)),
                    'thumbnailUri': os.path.join(db_prefix, scan.id,
                                                 "%s/thumbnail_%s.jpg" % (filesets_matches['Visualization'], f.id))})
                break

    return res


class ScanList(Resource):
    def get(self):
        query=request.args.get('filterQuery')
        scans=fmt_scans(db.get_scans(), query=query)
        return scans


class Scan(Resource):
    def get(self, scan_id):
        scan=db.get_scan(scan_id)
        filesets_matches=compute_fileset_matches(scan)
        return fmt_scan(scan, filesets_matches)

class File(Resource):
    def get(self, path):
        return send_from_directory(db_location, path)

class Refresh(Resource):
    def get(self):
        global db
        db=DB(db_location)
        db.connect()
        return 200



api.add_resource(ScanList, '/scans')
api.add_resource(Scan, '/scans/<scan_id>')
api.add_resource(File, '/files/<path:path>')
api.add_resource(Refresh, '/refresh')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
