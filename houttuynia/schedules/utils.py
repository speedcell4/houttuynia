__all__ = [
    'apply_arguments',
]


def apply_arguments(func, arguments):
    if isinstance(arguments, tuple) and hasattr(arguments, '_asdict'):
        return func(**arguments._asdict())
    if isinstance(arguments, (list, tuple)):
        return func(*arguments)
    if isinstance(arguments, dict):
        return func(**arguments)
    return func(arguments)
