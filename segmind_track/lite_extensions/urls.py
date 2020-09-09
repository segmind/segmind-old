import os

TRACKING_URI = os.environ.get('SEGMIND_TRACKING_URL',
                              'https://logs.segmind.com')
SEGMIND_API_URL = os.environ.get('SEGMIND_API_URL',
                                 'https://refine.segmind.com')
