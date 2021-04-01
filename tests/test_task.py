import unittest

import luigi
from plantdb import RomiTask, DatabaseConfig, FilesetTarget
from plantdb import io
from plantdb.task import FilesetExists, ImagesFilesetExists, FileByFileTask
from plantdb.testing import DBTestCase
from os import rmdir, path

class TouchFileTask(RomiTask):
    upstream_task = None

    def requires(self):
        return []

    def run(self):
        x = self.output().get()
        y = x.create_file("hello")
        y.write("hello", "txt")


class TestFilesetExists(FilesetExists):
    fileset_id = "testfileset"


class DoNothingTask(RomiTask):
    def requires(self):
        return TestFilesetExists()

    def run(self):
        pass


class ImageIdentityTask(FileByFileTask):
    reader = io.read_image
    writer = io.write_image
    upstream_task = None
    fileset_id = luigi.Parameter(default="images")

    def f(self, x, outfs):
        return x

    def requires(self):
        return ImagesFilesetExists(fileset_id=self.fileset_id)


class TestFilesetTarget(DBTestCase):
    def test_target(self):
        db = self.get_test_db()
        scan = db.get_scan("testscan")
        target = FilesetTarget(scan, "testfileset2")
        assert (target.get(create=False) is None)
        assert (not target.exists())
        target.create()
        assert (not target.exists())  # Target `Fileset` exist but is empty
        fs = scan.get_fileset("testfileset2")
        fs.create_file('dummy_test_file')  # Now target `Fileset` exist and is not empty
        assert (target.exists())
        assert (target.get() is not None)
        rmdir(path.join(target.scan.db.basedir, target.scan.id, target.fileset_id))


class TestRomiTask(DBTestCase):
    def test_romi_task(self):
        db = self.get_test_db()
        DatabaseConfig.db = db
        DatabaseConfig.scan = db.get_scan("testscan")
        task = TouchFileTask()
        assert (not task.complete())
        luigi.build(tasks=[task], local_scheduler=True)
        assert (task.complete())


class TestFileByFileTask(DBTestCase):
    def test_romi_task(self):
        db = self.get_test_db()
        DatabaseConfig.db = db
        DatabaseConfig.scan_id = "testscan"
        DatabaseConfig.scan = self.get_test_scan()
        # task = ImageIdentityTask(fileset_id="testfileset")
        # assert (not task.complete())
        # luigi.build(tasks=[task], local_scheduler=True)
        # assert (task.complete())
        luigi.build(tasks=[ImageIdentityTask(fileset_id="testfileset")], local_scheduler=True)


if __name__ == "__main__":
    unittest.main()
