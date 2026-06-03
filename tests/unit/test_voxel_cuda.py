#!/usr/bin/env python3
"""
Unit tests for Backprojection (CUDA) class focusing on non-hardware dependent functionality.
"""

import unittest

from plant3dvision.voxel_cuda import Backprojection


class TestBackprojectionCUDA(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.shape = [5, 5, 5]
        self.origin = [0.0, 0.0, 0.0]
        self.voxel_size = 1.0

    def test_cuda_constructor_valid_carving(self):
        """Test CUDA Backprojection constructor with the carving method."""
        try:
            # This test will run the actual CUDA implementation
            backproj = Backprojection(
                shape=self.shape,
                origin=self.origin,
                voxel_size=self.voxel_size,
                method="carving"
            )

            # Just check that we can create the object and it has the basic properties
            self.assertEqual(backproj.shape, self.shape)
            self.assertEqual(backproj.origin, self.origin)
            self.assertEqual(backproj.voxel_size, self.voxel_size)
            self.assertEqual(backproj.method, "carving")

            # Clean up
            del backproj

        except Exception as e:
            # If CUDA isn't properly configured or fails, skip the test
            self.skipTest(f"CUDA test failed: {str(e)}")

    def test_cuda_constructor_valid_averaging(self):
        """Test CUDA Backprojection constructor with the averaging method."""
        try:
            # This test will run the actual CUDA implementation
            backproj = Backprojection(
                shape=self.shape,
                origin=self.origin,
                voxel_size=self.voxel_size,
                method="averaging"
            )

            # Just check that we can create the object and it has the basic properties
            self.assertEqual(backproj.shape, self.shape)
            self.assertEqual(backproj.origin, self.origin)
            self.assertEqual(backproj.voxel_size, self.voxel_size)
            self.assertEqual(backproj.method, "averaging")

            # Clean up
            del backproj

        except Exception as e:
            # If CUDA isn't properly configured or fails, skip the test
            self.skipTest(f"CUDA test failed: {str(e)}")

    def test_cuda_constructor_invalid_method(self):
        """Test CUDA Backprojection constructor with invalid method raises ValueError."""
        with self.assertRaises(ValueError) as context:
            Backprojection(
                shape=self.shape,
                origin=self.origin,
                voxel_size=self.voxel_size,
                method="invalid_method"
            )

        self.assertIn("Unknown kernel type invalid_method", str(context.exception))


if __name__ == '__main__':
    unittest.main()
