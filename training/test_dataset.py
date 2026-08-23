import unittest

from dataset import chunk_text


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [int(value) for value in text.split()]

    def decode(self, values, skip_special_tokens=False):
        return " ".join(str(value) for value in values)


class ChunkTextTests(unittest.TestCase):
    def test_rebalances_small_tail(self):
        chunks = chunk_text(" ".join(map(str, range(11))), FakeTokenizer(), 10, 3)
        self.assertEqual([len(chunk.split()) for chunk in chunks], [6, 5])

    def test_does_not_split_short_example(self):
        text = "1 2 3"
        self.assertEqual(chunk_text(text, FakeTokenizer(), 10, 3), [text])


if __name__ == "__main__":
    unittest.main()
