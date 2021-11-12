import os

TRACKING_URI = os.environ.get('SEGMIND_TRACKING_URL',
                              'https://cloud.segmind.com/track')
SEGMIND_API_URL = os.environ.get('SEGMIND_API_URL',
                                 'https://cloud.segmind.com/api')
SEGMIND_SPOT_URL = os.environ.get('SEGMIND_SPOT_URL',
                                  'https://api.spotprod.segmind.com')
