import unittest

import numpy as np

# Import only the abstract base class which should work without external dependencies
from plant3dvision.voxel import AbstractBackprojection


class TestAbstractBackprojection(unittest.TestCase):
    """Test suite for AbstractBackprojection class functionality."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.shape = [10, 10, 10]
        self.origin = [0.0, 0.0, 0.0]
        self.voxel_size = 1.0

    def test_constructor_valid_carving(self):
        """Test AbstractBackprojection constructor with valid carving method."""

        # Create a concrete implementation for testing
        class TestBackprojection(AbstractBackprojection):
            def init_buffers(self):
                pass

            def process_view(self, intrinsics, rot, tvec, mask):
                pass

            def get_values(self):
                return np.zeros(self.shape, dtype=self.dtype)

            def clear(self):
                pass

        # Arrange & Act
        backproj = TestBackprojection(
            shape=self.shape,
            origin=self.origin,
            voxel_size=self.voxel_size,
            method="carving"
        )

        # Assert
        self.assertEqual(backproj.shape, self.shape)
        self.assertEqual(backproj.origin, self.origin)
        self.assertEqual(backproj.voxel_size, self.voxel_size)
        self.assertEqual(backproj.method, "carving")
        self.assertEqual(backproj.dtype, np.int32)
        self.assertEqual(backproj.default_value, 0.0)
        self.assertFalse(backproj.log)

    def test_constructor_valid_averaging(self):
        """Test AbstractBackprojection constructor with valid averaging method."""

        # Create a concrete implementation for testing
        class TestBackprojection(AbstractBackprojection):
            def init_buffers(self):
                pass

            def process_view(self, intrinsics, rot, tvec, mask):
                pass

            def get_values(self):
                return np.zeros(self.shape, dtype=self.dtype)

            def clear(self):
                pass

        # Arrange & Act
        backproj = TestBackprojection(
            shape=self.shape,
            origin=self.origin,
            voxel_size=self.voxel_size,
            method="averaging"
        )

        # Assert
        self.assertEqual(backproj.shape, self.shape)
        self.assertEqual(backproj.origin, self.origin)
        self.assertEqual(backproj.voxel_size, self.voxel_size)
        self.assertEqual(backproj.method, "averaging")
        self.assertEqual(backproj.dtype, np.float32)
        self.assertEqual(backproj.default_value, 0.0)
        self.assertFalse(backproj.log)

    def test_constructor_invalid_method_raises_error(self):
        """Test AbstractBackprojection constructor with invalid method raises ValueError."""

        # Create a concrete implementation for testing
        class TestBackprojection(AbstractBackprojection):
            def init_buffers(self):
                pass

            def process_view(self, intrinsics, rot, tvec, mask):
                pass

            def get_values(self):
                return np.zeros(self.shape, dtype=self.dtype)

            def clear(self):
                pass

        # Arrange, Act & Assert
        with self.assertRaises(ValueError) as context:
            TestBackprojection(
                shape=self.shape,
                origin=self.origin,
                voxel_size=self.voxel_size,
                method="invalid_method"
            )

        self.assertIn("Unknown kernel type invalid_method", str(context.exception))

    def test_prepare_mask_float32_no_log(self):
        """Test _prepare_mask method with float32 dtype and log=False."""

        # Create a concrete implementation for testing
        class TestBackprojection(AbstractBackprojection):
            def init_buffers(self):
                pass

            def process_view(self, intrinsics, rot, tvec, mask):
                pass

            def get_values(self):
                return np.zeros(self.shape, dtype=self.dtype)

            def clear(self):
                pass

        # Arrange
        backproj = TestBackprojection(
            shape=self.shape,
            origin=self.origin,
            voxel_size=self.voxel_size,
            method="averaging"
        )

        # Create a test mask
        mask = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

        # Act
        result = backproj._prepare_mask(mask)

        # Assert
        # Should convert to float32 but not apply log transform
        self.assertEqual(result.dtype, np.float32)
        # The actual conversion should be using img_as_float32 from skimage
        # But for our test we just verify the dtype conversion happened
        self.assertEqual(result.shape, mask.shape)

    def test_prepare_mask_float32_with_log(self):
        """Test _prepare_mask method with float32 dtype and log=True."""

        # Create a concrete implementation for testing
        class TestBackprojection(AbstractBackprojection):
            def init_buffers(self):
                pass

            def process_view(self, intrinsics, rot, tvec, mask):
                pass

            def get_values(self):
                return np.zeros(self.shape, dtype=self.dtype)

            def clear(self):
                pass

        # Arrange
        backproj = TestBackprojection(
            shape=self.shape,
            origin=self.origin,
            voxel_size=self.voxel_size,
            method="averaging",
            log=True
        )

        # Create a test mask
        mask = np.array([[1, 2, 1], [2, 1, 2], [1, 2, 1]], dtype=np.uint8)

        # Act
        result = backproj._prepare_mask(mask)

        # Assert
        # Should convert to float32 and apply log transform
        self.assertEqual(result.dtype, np.float32)
        # Check that the result has the same shape as input
        self.assertEqual(result.shape, mask.shape)

    def test_validate_mask_valid(self):
        """Test _validate_mask method with valid mask."""

        # Create a concrete implementation for testing
        class TestBackprojection(AbstractBackprojection):
            def init_buffers(self):
                pass

            def process_view(self, intrinsics, rot, tvec, mask):
                pass

            def get_values(self):
                return np.zeros(self.shape, dtype=self.dtype)

            def clear(self):
                pass

        # Arrange
        backproj = TestBackprojection(
            shape=self.shape,
            origin=self.origin,
            voxel_size=self.voxel_size,
            method="carving"
        )

        # Create a valid test mask
        mask = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

        # Act
        result = backproj._validate_mask(mask)

        # Assert
        self.assertTrue(result)

    def test_validate_mask_empty(self):
        """Test _validate_mask method with empty mask."""

        # Create a concrete implementation for testing
        class TestBackprojection(AbstractBackprojection):
            def init_buffers(self):
                pass

            def process_view(self, intrinsics, rot, tvec, mask):
                pass

            def get_values(self):
                return np.zeros(self.shape, dtype=self.dtype)

            def clear(self):
                pass

        # Arrange
        backproj = TestBackprojection(
            shape=self.shape,
            origin=self.origin,
            voxel_size=self.voxel_size,
            method="carving"
        )

        # Create an empty test mask
        mask = np.array([], dtype=np.uint8)

        # Act
        result = backproj._validate_mask(mask)

        # Assert
        self.assertFalse(result)

    def test_constructor_dtype_assignment(self):
        """Test correct dtype assignment for carving vs averaging."""

        # Create a concrete implementation for testing
        class TestBackprojection(AbstractBackprojection):
            def init_buffers(self):
                pass

            def process_view(self, intrinsics, rot, tvec, mask):
                pass

            def get_values(self):
                return np.zeros(self.shape, dtype=self.dtype)

            def clear(self):
                pass

        # Carving method
        bp_carving = TestBackprojection(
            shape=self.shape,
            origin=self.origin,
            voxel_size=self.voxel_size,
            method="carving"
        )
        self.assertEqual(bp_carving.dtype, np.int32)

        # Averaging method
        bp_averaging = TestBackprojection(
            shape=self.shape,
            origin=self.origin,
            voxel_size=self.voxel_size,
            method="averaging"
        )
        self.assertEqual(bp_averaging.dtype, np.float32)

    def test_constructor_default_values(self):
        """Test default values assignment."""

        # Create a concrete implementation for testing
        class TestBackprojection(AbstractBackprojection):
            def init_buffers(self):
                pass

            def process_view(self, intrinsics, rot, tvec, mask):
                pass

            def get_values(self):
                return np.zeros(self.shape, dtype=self.dtype)

            def clear(self):
                pass

        bp = TestBackprojection(
            shape=self.shape,
            origin=self.origin,
            voxel_size=self.voxel_size,
            method="carving",
            default_value=5.0,
            log=True
        )

        self.assertEqual(bp.default_value, 5.0)
        self.assertTrue(bp.log)


if __name__ == '__main__':
    unittest.main()
