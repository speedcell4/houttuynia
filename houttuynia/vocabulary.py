from collections import Counter
from pathlib import Path
from typing import Iterable, Tuple, Union
import functools

from tqdm import tqdm

__all__ = [
    'Vocab',
]


def built_required(method):
    @functools.wraps(method)
    def wrap(self, *args, **kwargs):
        if self._built:
            return method(self, *args, **kwargs)
        raise RuntimeError(f'{self.__class__.__name__} should be built first')

    return wrap


class Vocab(object):
    unk: int
    pad: int

    def __init__(self, counter: Counter = None, specials: Tuple[str, ...] = ('<unk>', '<pad>')) -> None:
        super(Vocab, self).__init__()

        if counter is None:
            counter = Counter()

        for spc in specials:
            if not spc.startswith('<') or not spc.endswith('>'):
                raise ValueError(f'{spc} is not an valid special token')

        self._built = False
        self.counter = counter
        self.specials = specials

    def update(self, iteration: Iterable[str]) -> None:
        self._built = False
        return self.counter.update(iteration)

    def build_vocab(self, max_vocab_size: int = None, min_freq: int = 1) -> None:
        self._token_ix = {}
        self._ix_token = {}
        for ix, (token, freq) in tqdm(
                enumerate(self.__iter__()), desc=f'{self.__class__.__name__}', unit=' tokens'):
            if freq < min_freq:
                break
            if max_vocab_size is not None and max_vocab_size <= ix:
                break
            self._token_ix[token] = ix
            self._ix_token[ix] = token

        for sp in self.specials:
            setattr(self, sp[1:-1], self._token_ix[sp])

        self._built = True

    def dump(self, file: Union[str, Path], encoding: str = 'utf-8') -> None:
        with Path(file).open(mode='w', encoding=encoding) as fp:
            for token, freq in self.__iter__():
                fp.write(f'{token}\t{freq}\n')

    @classmethod
    def load(cls, file: Union[str, Path], encoding: str = 'utf-8') -> 'Vocab':
        counter, reserves = Counter(), []
        with Path(file).open(mode='r', encoding=encoding) as fp:
            for raw_line in fp:
                token, freq = raw_line.strip().split('\t')
                if freq == 'inf':
                    reserves.append(token)
                else:
                    counter[token] = int(freq)
        return Vocab(counter, tuple(reserves))

    @property
    def built(self) -> bool:
        return self._built

    @property
    @built_required
    def tokens(self) -> Tuple[str, ...]:
        return tuple(self._token_ix.keys())

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

    @built_required
    def __contains__(self, token: str) -> bool:
        return token in self._token_ix

    @built_required
    def __len__(self) -> int:
        return self._token_ix.__len__()

    @built_required
    def __call__(self, item: Union[int, Iterable[str]]) -> Union[int, Iterable[int]]:
        if isinstance(item, int):
            raise TypeError(f'use {self.__class__.__name__}.__getitem__ instead')
        if isinstance(item, str):
            return self._token_ix.get(item, self.unk)
        return type(item)(self(item) for item in item)

    @built_required
    def __getitem__(self, item: Union[int, Iterable[int]]) -> Union[str, Iterable[str]]:
        if isinstance(item, str):
            raise TypeError(f'use {self.__class__.__name__}.__call__ instead')
        if isinstance(item, int):
            return self._ix_token[item]
        return type(item)(self[item] for item in item)
