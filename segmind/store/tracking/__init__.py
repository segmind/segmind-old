"""An MLflow tracking server has two properties related to how data is stored:

*backend store* to record ML experiments, runs, parameters, metrics, etc., and
*artifact store* to store run artifacts like models, plots, images, etc.

Several constants are used by multiple backend store implementations.
"""

# Path to default location for backend when using local FileStore or ArtifactStore.
# Also used as default location for artifacts, when not provided, in non local file based backends
# (eg MySQL)

from segmind.lite_extensions.server_only.file_utils import _ROOT_DIR

DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH = _ROOT_DIR  #"./mlruns"
SEARCH_MAX_RESULTS_DEFAULT = 1000
SEARCH_MAX_RESULTS_THRESHOLD = 50000
