import luigi

from lettucethink.db.fsdb import DB

class DatabaseConfig(luigi.Config):
    db_location = luigi.Parameter()
    scan_id = luigi.Parameter()


class FilesetTarget(luigi.Target):
    def __init__(self, db_location, scan_id, fileset_id):
        db = DB(db_location)
        db.connect()
        scan = db.get_scan(scan_id)
        if scan is None:
            raise Exception("Scan does not exist")
        self.scan = scan
        self.fileset_id = fileset_id

    def create(self):
        return self.scan.create_fileset(self.fileset_id)

    def exists(self):
        fs = self.scan.get_fileset(self.fileset_id)
        return fs is not None and len(fs.get_files()) > 0

    def get(self, create=True):
        return self.scan.get_fileset(self.fileset_id, create=create)

class RomiTask(luigi.Task):
    def output(self):
        fileset_id = self.task_id
        return FilesetTarget(DatabaseConfig().db_location, DatabaseConfig().scan_id, fileset_id)

    def complete(self):
        outs = self.output()
        if isinstance(outs, dict):
            outs = [outs[k] for k in outs.keys()]
        elif isinstance(outs, list):
            pass
        else:
            outs = [outs]

        if not all(map(lambda output: output.exists(), outs)):
            return False

        req = self.requires()
        if isinstance(req, dict):
            req = [req[k] for k in req.keys()]
        elif isinstance(req, list):
            pass
        else:
            req = [req]
        for task in req:
            if not task.complete():
                return False
        return True

@RomiTask.event_handler(luigi.Event.FAILURE)
def mourn_failure(task, exception):
    output_fileset = task.output().get()
    scan = task.output().get().scan
    scan.delete_fileset(output_fileset.id)
