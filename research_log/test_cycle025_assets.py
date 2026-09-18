import unittest
import numpy as np
from prepare_cycle025 import mask_reason
class AssetTests(unittest.TestCase):
 def test_empty_target_fails(self):self.assertEqual(mask_reason(np.zeros((4,5)),(4,5)),'empty_mask')
 def test_wrong_size_fails(self):self.assertEqual(mask_reason(np.ones((5,4)),(4,5)),'mask_not_image_aligned')
 def test_nonempty_aligned_mask_passes(self):self.assertIsNone(mask_reason(np.eye(4),(4,4)))
if __name__=='__main__':unittest.main()
