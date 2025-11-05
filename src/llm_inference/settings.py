import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CACHE_DIR = os.path.join(BASE_DIR, '..', '..', '.cache')

LOGGING_LEVEL = os.getenv('LOGGING_LEVEL', 'INFO')