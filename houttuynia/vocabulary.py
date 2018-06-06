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

    def __init__(self, counter: Counter = None, reserves: Tuple[str, ...] = ('<unk>', '<pad>')) -> None:
        super(Vocab, self).__init__()

        if counter is None:
            counter = Counter()

        for reserve in reserves:
            if not reserve.startswith('<') or not reserve.endswith('>'):
                raise ValueError(f'{reserve} is not an expected format')

        self.counter = counter
        self.reserves = reserves

    def update(self, iteration: Iterable[str]) -> None:
        return self.counter.update(iteration)

    def build_vocab(self, max_vocab_size: int = None, min_freq: int = 1) -> None:
        self._token_ix = {}
        self._ix_token = {}
        for ix, (token, freq) in tqdm(enumerate(self.__iter__()), desc='building vocabulary', unit=' token'):
            if freq < min_freq:
                break
            if max_vocab_size is not None and max_vocab_size <= ix:
                break
            self._token_ix[token] = ix
            self._ix_token[ix] = token

        for reserve in self.reserves:
            setattr(self, reserve[1:-1], self._token_ix[reserve])

    @classmethod
    def load(cls, file: Union[str, Path], encoding: str = 'utf-8') -> 'Vocab':
        counter, reserves = Counter(), []
        with Path(file).open(mode='r', encoding=encoding) as fp:
            for raw_line in fp:
                token, freq = raw_line.strip().split(' ')
                if freq == 'inf':
                    reserves.append(token)
                else:
                    counter[token] = int(freq)
        return Vocab(counter, tuple(reserves))

    def dump(self, file: Union[str, Path], encoding: str = 'utf-8') -> None:
        with Path(file).open(mode='w', encoding=encoding) as fp:
            for token, freq in self.__iter__():
                fp.write(f'{token} {freq}\n')

    @property
    def tokens(self) -> KeysView[str]:
        return self._token_ix.keys()

    def __iter__(self) -> Iterable[Tuple[str, Union[int, float]]]:
        for token in self.reserves:
            yield token, float('inf')
        for token, freq in self.counter.most_common():
            yield token, freq

    def __add__(self, other: 'Vocab') -> 'Vocab':
        return Vocab(counter=self.counter + other.counter,
                     reserves=tuple(set(self.reserves + other.reserves)))

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}, {self.__len__()} tokens>'

    def __contains__(self, token: str) -> bool:
        return token in self._token_ix

    def __len__(self) -> int:
        return self._token_ix.__len__()

    def __getitem__(self, index: int) -> str:
        return self._ix_token[index]

    def __call__(self, token: str) -> int:
        return self._token_ix.get(token, self.unk)


if __name__ == '__main__':
    vocab = Vocab.load(Path('nice.vocab'))
    vocab.build_vocab()
    print(vocab.tokens)
