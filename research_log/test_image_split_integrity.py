import unittest
from check_image_split_integrity import audit_roles

class SplitIntegrityTests(unittest.TestCase):
 def test_instance_alias_crossing_roles_fails(self):
  rows=[{'image_sha256':'a'*64,'role':'train','path':'ann1.jpg'}, {'image_sha256':'a'*64,'role':'holdout','path':'ann2.jpg'}]
  self.assertEqual(audit_roles(rows)['cross_role_images'],1)
 def test_repeated_identity_within_role_passes(self):
  rows=[{'image_sha256':'a'*64,'role':'train'}]*2+[{'image_sha256':'b'*64,'role':'holdout'}]
  self.assertEqual(audit_roles(rows)['status'],'PASS')
if __name__=='__main__':unittest.main()
