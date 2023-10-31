import unittest

from biogeme.partition import Partition


class TestPartition(unittest.TestCase):
    def test_valid_partition(self):
        """Test a valid partition."""
        segments = [{1, 2}, {3, 4, 5}]
        full_set = {1, 2, 3, 4, 5}
        partition = Partition(segments, full_set)
        self.assertEqual(list(partition), segments)

    def test_invalid_segment_type(self):
        """Test for a segment that is not of type 'set'."""
        segments = [[1, 2], {3, 4, 5}]
        full_set = {1, 2, 3, 4, 5}
        with self.assertRaises(ValueError) as context:
            Partition(segments, full_set)
        self.assertIn('Each segment must be of type "set".', str(context.exception))

    def test_invalid_segment_item_type(self):
        """Test for a segment item that is not of type 'int'."""
        segments = [{1, '2'}, {3, 4, 5}]
        full_set = {1, 2, 3, 4, 5}
        with self.assertRaises(ValueError) as context:
            Partition(segments, full_set)
        self.assertIn(
            'Each item in the segment must be an integer.', str(context.exception)
        )

    def test_intersection_between_segments(self):
        """Test for segments that have an intersection."""
        segments = [{1, 2, 3}, {3, 4, 5}]
        full_set = {1, 2, 3, 4, 5}
        with self.assertRaises(ValueError) as context:
            Partition(segments, full_set)
        self.assertIn('Elements in two different segments:', str(context.exception))

    def test_union_does_not_match_full_set(self):
        """Test for segments whose union does not match the full set."""
        segments = [{1, 2}, {4, 5}]
        full_set = {1, 2, 3, 4, 5}
        with self.assertRaises(ValueError) as context:
            Partition(segments, full_set)
        self.assertIn(
            'The union of segments does not match the given set.',
            str(context.exception),
        )
        self.assertIn('Missing elements:', str(context.exception))

    def test_iterator(self):
        """Test the iterator of the Partition class."""
        segments = [{1, 2}, {3, 4, 5}]
        full_set = {1, 2, 3, 4, 5}
        partition = Partition(segments, full_set)

        # Verify the order and content of the yielded segments
        for expected_segment, yielded_segment in zip(segments, partition):
            self.assertEqual(expected_segment, yielded_segment)


if __name__ == '__main__':
    unittest.main()
