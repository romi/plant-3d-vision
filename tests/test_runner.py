import unittest

from plantdb.runner import DBRunner
from plantdb.testing import DBTestCase


class TestFSDBRunner(DBTestCase):
    def test_run_scan(self):
        db = self.get_test_db()
        runner = DBRunner(db, [], {})
        runner.run_scan("testscan")

    def test_run(self):
        db = self.get_test_db()
        runner = DBRunner(db, [], {})
        runner.run()


if __name__ == "__main__":
    unittest.main()
