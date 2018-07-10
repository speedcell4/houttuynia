from typing import List

from hypothesis import strategies as st
from hypothesis.searchstrategy import SearchStrategy

VOCAB_SIZE = st.integers(20, 1000)

BATCH = st.integers(1, 20)
TIMES = st.integers(1, 50)

TINY_FEATURE = st.integers(5, 20)
SMALL_FEATURE = st.integers(20, 100)
NORMAL_FEATURE = st.integers(100, 200)
LARGE_FEATURE = st.integers(200, 500)


def batched_lens(batch: int, min_len: int, max_len: int) -> List[int]:
    return st.lists(st.integers(min_len, max_len), min_size=batch, max_size=batch).example()


def sequences_by_batched_lens(token: SearchStrategy[int], batched_sizes: List[int]) -> List[List[int]]:
    return [st.lists(token, min_size=batched_size, max_size=batched_size).example() for batched_size in batched_sizes]


def batched_sequences(token: SearchStrategy[int], batch: int, min_len: int, max_len: int) -> List[List[int]]:
    batched_sizes = batched_lens(batch, min_len, max_len)
    return sequences_by_batched_lens(token, batched_sizes)
