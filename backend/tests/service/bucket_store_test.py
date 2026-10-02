import unittest

from src.storage import bucket_store
from tests.support.fake_bucket import FakeS3Client


class TestBucketStore(unittest.TestCase):
    def setUp(self):
        self.fake = FakeS3Client()
        bucket_store.set_client_for_tests(self.fake, "test-bucket")

    def tearDown(self):
        bucket_store.reset_client_for_tests()

    def test_put_and_get_bytes_roundtrip(self):
        bucket_store.put_bytes("a/b.txt", b"hello")
        self.assertEqual(bucket_store.get_bytes("a/b.txt"), b"hello")

    def test_get_bytes_missing_key_raises(self):
        with self.assertRaises(bucket_store.BucketKeyNotFound):
            bucket_store.get_bytes("missing")

    def test_exists(self):
        self.assertFalse(bucket_store.exists("x.json"))
        bucket_store.put_json("x.json", {"a": 1})
        self.assertTrue(bucket_store.exists("x.json"))

    def test_put_and_get_json_roundtrip(self):
        payload = {"market": "h2h", "stage": "production", "nested": [1, 2, 3]}
        bucket_store.put_json("registry/runs/run-1.json", payload)
        self.assertEqual(bucket_store.get_json("registry/runs/run-1.json"), payload)

    def test_get_json_missing_key_returns_default(self):
        self.assertEqual(bucket_store.get_json("missing.json", default={}), {})
        self.assertIsNone(bucket_store.get_json("missing.json"))

    def test_get_json_corrupt_content_returns_default(self):
        bucket_store.put_bytes("bad.json", b"not valid json {{{")
        self.assertEqual(bucket_store.get_json("bad.json", default=[]), [])

    def test_put_and_get_joblib_roundtrip(self):
        obj = {"weights": [0.1, 0.2, 0.3], "name": "fake_model"}
        bucket_store.put_joblib("models/fake.pkl", obj)
        loaded = bucket_store.get_joblib("models/fake.pkl")
        self.assertEqual(loaded, obj)

    def test_list_keys_filters_by_prefix(self):
        bucket_store.put_json("registry/runs/a.json", {})
        bucket_store.put_json("registry/runs/b.json", {})
        bucket_store.put_json("registry/promotions/c.json", {})

        run_keys = bucket_store.list_keys("registry/runs/")
        self.assertEqual(sorted(run_keys), ["registry/runs/a.json", "registry/runs/b.json"])

    def test_list_keys_empty_prefix_returns_empty(self):
        self.assertEqual(bucket_store.list_keys("nothing/here/"), [])

    def test_delete_removes_key(self):
        bucket_store.put_json("x.json", {"a": 1})
        bucket_store.delete("x.json")
        self.assertFalse(bucket_store.exists("x.json"))


if __name__ == "__main__":
    unittest.main()
