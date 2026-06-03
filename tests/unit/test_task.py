import unittest
from os import path
from os import rmdir

import luigi

from plantdb.commons import io
from plantdb.commons.fsdb.exceptions import FilesetNotFoundError
from plantdb.commons.testing import DummyDBTestCase
from plantdb.commons.testing import FSDBTestCase
from romitask import FilesetTarget
from romitask import RomiTask
from romitask import ScanConfiguration
from romitask.task import FileByFileTask
from romitask.task import FilesetExists
from romitask.task import ImagesFilesetExists


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


class TestFilesetTarget(DummyDBTestCase):
    def test_target(self):
        scan = self.db.get_scan("myscan_001")
        target = FilesetTarget(scan, "testfileset2")
        with self.assertRaises(FilesetNotFoundError):
            target.get()
        assert (not target.exists())
        target.create()
        assert (not target.exists())  # Target `Fileset` exist but is empty
        assert "testfileset2" in scan.list_filesets()
        fs = scan.get_fileset("testfileset2")
        fs.create_file('dummy_test_file')  # Now target `Fileset` exist and is not empty
        assert (target.exists())
        assert (target.get() is not None)
        rmdir(path.join(target.scan.db.basedir, target.scan.id, target.fileset_id))


class TestRomiTask(DummyDBTestCase):
    def test_romi_task(self):
        ScanConfiguration.db = self.db
        ScanConfiguration.scan = self.db.get_scan("myscan_001")
        task = TouchFileTask()
        assert (not task.complete())
        luigi.build(tasks=[task], local_scheduler=True)
        assert (task.complete())


class TestFileByFileTask(FSDBTestCase):
    def test_romi_task(self):
        db = self.get_test_db()
        ScanConfiguration.db = db
        ScanConfiguration.scan_id = "myscan_001"
        ScanConfiguration.scan = self.get_test_scan()
        # task = ImageIdentityTask(fileset_id="testfileset")
        # assert (not task.complete())
        # luigi.build(tasks=[task], local_scheduler=True)
        # assert (task.complete())
        luigi.build(tasks=[ImageIdentityTask(fileset_id="testfileset")], local_scheduler=True)


if __name__ == "__main__":
    unittest.main()
