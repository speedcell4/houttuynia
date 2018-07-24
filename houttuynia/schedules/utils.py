__all__ = [
    'apply_batch',
]


def apply_batch(func, batch):
    if isinstance(batch, (list, tuple)):
        return func(*batch)
    if isinstance(batch, dict):
        return func(**batch)
    return func(batch)
