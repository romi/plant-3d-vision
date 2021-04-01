import os
import unittest

from plantdb.testing import DBTestCase
from plantdb.watch import FSDBWatcher


# class TouchFileTask(RomiTask):
#     def requires(self):
#         return[]
#     def run(self):
#         x = self.output().get()
#         y = x.create_file("hello")
#         y.write_text("txt", "hello")


class TestFSDBWatcher(DBTestCase):
    def test_watch(self):
        db = self.get_test_db()
        watcher = FSDBWatcher(db, [], {})
        watcher.start()
        os.makedirs(os.path.join(db.basedir, "test"))
        watcher.stop()
        watcher.join()


if __name__ == "__main__":
    unittest.main()
