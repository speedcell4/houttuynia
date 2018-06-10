from functools import wraps

from houttuynia import config


# TODO unwrap with name, not only chapter
def unwrap_chapter(func):
    @wraps(func)
    def wrapper(*args, chapter=None, **kwargs):
        if chapter is None:
            chapter = config['chapter']

        return func(*args, chapter=chapter, **kwargs)

    return wrapper
