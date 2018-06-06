from collections import Counter
from pathlib import Path
from typing import Iterable, KeysView, Tuple, Union

from tqdm import tqdm

__all__ = [
    'Vocab',
]


class Vocab(object):
    unk: int
    pad: int

    def __init__(self, counter: Counter = None, specials: Tuple[str, ...] = ('<unk>', '<pad>')) -> None:
        super(Vocab, self).__init__()

        if counter is None:
            counter = Counter()

        for special in specials:
            if not special.startswith('<') or not special.endswith('>'):
                raise ValueError(f'{special} is not an expected format')

        self.counter = counter
        self.specials = specials

    def update(self, iteration: Iterable[str]) -> None:
        return self.counter.update(iteration)

    def build_vocab(self, max_vocab_size: int = None, min_freq: int = 1) -> None:
        self._stoi = {}
        self._itos = {}
        for ix, (token, freq) in tqdm(enumerate(self.__iter__()), desc='building vocabulary', unit=' token'):
            if freq < min_freq:
                break
            if max_vocab_size is not None and max_vocab_size <= ix:
                break
            self._stoi[token] = ix
            self._itos[ix] = token

        for special in self.specials:
            setattr(self, special[1:-1], self._stoi[special])

    @classmethod
    def load(cls, file: Path, encoding: str = 'utf-8') -> 'Vocab':
        counter, specials = Counter(), []
        with file.open(mode='r', encoding=encoding) as fp:
            for raw_line in fp:
                token, freq = raw_line.strip().split(' ')
                if freq == 'inf':
                    specials.append(token)
                else:
                    counter[token] = int(freq)
        return Vocab(counter, tuple(specials))

    def dump(self, file: Path, encoding: str = 'utf-8') -> None:
        with file.open(mode='w', encoding=encoding) as fp:
            for token, freq in self.__iter__():
                fp.write(f'{token} {freq}\n')

    @property
    def tokens(self) -> KeysView[str]:
        return self._stoi.keys()

    def __iter__(self) -> Iterable[Tuple[str, Union[int, float]]]:
        for token in self.specials:
            yield token, float('inf')
        for token, freq in self.counter.most_common():
            yield token, freq

    def __add__(self, other: 'Vocab') -> 'Vocab':
        return Vocab(counter=self.counter + other.counter,
                     specials=tuple(set(self.specials + other.specials)))

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}, {self.__len__()} tokens>'

    def __contains__(self, token: str) -> bool:
        return token in self._stoi

    def __len__(self) -> int:
        return self._stoi.__len__()

    def __getitem__(self, index: int) -> str:
        return self._itos[index]

    def __call__(self, token: str) -> int:
        return self._stoi.get(token, self.unk)


if __name__ == '__main__':
    vocab = Vocab.load(Path('nice.vocab'))
    vocab.build_vocab()
    print(vocab.tokens)
