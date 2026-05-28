import unittest

from plantdb.commons.testing import FSDBTestCase
from romitask.runner import DBRunner


class TestFSDBRunner(FSDBTestCase):

    def test_run_scan(self):
        db = self.get_test_db()
        runner = DBRunner(db, [], {})
        runner.run_scan("real_plant_analyzed")

    def test_run(self):
        db = self.get_test_db()
        runner = DBRunner(db, [], {})
        runner.run()


if __name__ == "__main__":
    unittest.main()
