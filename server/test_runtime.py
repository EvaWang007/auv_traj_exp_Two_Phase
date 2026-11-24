"""CPU-only tests for the environment gate; no training or CUDA allocation."""
import unittest
from check_runtime import validate_runtime


class RuntimeTests(unittest.TestCase):
    def test_rented_image(self):
        validate_runtime((3,12,10),'2.8.0+cu128','12.8',True)

    def test_image_without_version_suffix(self):
        validate_runtime((3,12,1),'2.8.0','12.8',True)

    def test_legacy(self):
        validate_runtime((3,10,20),'2.7.1+cu118','11.8',True)

    def test_wrong_runtime_rejected(self):
        with self.assertRaises(RuntimeError): validate_runtime((3,12,1),'2.8.0+cu126','12.6',True)

    def test_no_gpu_rejected(self):
        with self.assertRaises(RuntimeError): validate_runtime((3,12,1),'2.8.0+cu128','12.8',False)


if __name__=='__main__': unittest.main(verbosity=2)
