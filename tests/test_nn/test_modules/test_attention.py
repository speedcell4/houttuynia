import unittest

import torch
from hypothesis import given, strategies as st

from houttuynia.nn.modules.utils import lens_to_mask
from houttuynia.nn.modules.attention import FacetAttention, MultiHeadAttention
import houttuynia as ho
from houttuynia import testing


@st.composite
def attention_st(
        draw,
        batch=testing.SMALL_BATCH, use_mask=st.booleans(),
        nom1=testing.SMALL_BATCH, nom2=testing.SMALL_BATCH,
        key_features=testing.NORMAL_FEATURE, value_features=testing.NORMAL_FEATURE):
    use_mask = draw(use_mask)

    batch = draw(batch)
    nom1, nom2 = draw(nom1), draw(nom2)
    key_features = draw(key_features)
    value_features = draw(value_features)

    Q = torch.rand(batch, nom1, key_features)
    K = torch.rand(batch, nom2, key_features)
    V = torch.rand(batch, nom2, value_features)

    if use_mask:
        lens = draw(st.lists(st.integers(1, nom2), min_size=batch, max_size=batch))
        mask = lens_to_mask(ho.long_tensor(lens), total_length=nom2, batch_first=True)
    else:
        mask = None

    del draw
    return locals()


class TestFacetAttention(unittest.TestCase):
    @given(data=attention_st())
    def test_shape(self, data):
        attention = FacetAttention(data['key_features'])
        outputs = attention(data['Q'], data['K'], data['V'], data['mask'])
        self.assertIs(outputs.dtype, torch.float32)
        self.assertEqual(outputs.size(), (data['batch'], data['nom1'], data['value_features']))


class TestMultiHeadAttention(unittest.TestCase):
    @given(
        data=attention_st(),
        num_heads=st.integers(1, 10),
        model_features=testing.SMALL_FEATURE,
    )
    def test_shape(self, data, num_heads, model_features):
        attention = MultiHeadAttention(
            num_heads=num_heads,
            key_features=data['key_features'],
            value_features=data['value_features'],
            out_features=num_heads * model_features,
        )
        outputs = attention(data['Q'], data['K'], data['V'], data['mask'])
        self.assertIs(outputs.dtype, torch.float32)
        self.assertEqual(outputs.size(), (data['batch'], data['nom1'], num_heads * model_features))
