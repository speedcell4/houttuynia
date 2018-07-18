from unittest import TestCase

from houttuynia.vocabulary import Vocab


class TestVocab(TestCase):
    def setUp(self):
        self.vocab = Vocab()
        self.vocab.update('hello work'.split())
        self.vocab.update('this is a nice framework , very nice'.split())

    def tearDown(self):
        del self.vocab

    def test_build_vocab(self):
        self.vocab.build_vocab()
        expected_tokens = ('<unk>', '<pad>', 'nice',
                           'hello', 'work', 'this',
                           'is', 'a', 'framework', ',', 'very')
        self.assertEqual(self.vocab.tokens, expected_tokens)

    def test_call(self):
        with self.assertRaises(RuntimeError):
            _ = self.vocab(1)
        self.vocab.build_vocab()
        expected_indexes = [0, 9, 2, 4, 0, 6]
        self.assertEqual(self.vocab('yes , nice work it is'.split()), expected_indexes)

    def test_getitem(self):
        with self.assertRaises(RuntimeError):
            _ = self.vocab['a']
        self.vocab.build_vocab()
        expected_tokens = tuple('<unk> , nice work <unk> is'.split())
        self.assertEqual(self.vocab[0, 9, 2, 4, 0, 6], expected_tokens)

    def test_len(self):
        with self.assertRaises(RuntimeError):
            _ = self.vocab.__len__()
        self.vocab.build_vocab()
        self.assertEqual(self.vocab.__len__(), 11)
