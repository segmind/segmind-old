"""File containing all of the run tags in the cral.tracking. namespace.

See the REST API documentation for information on the meaning of these tags.
"""

MLFLOW_RUN_NAME = 'cral.tracking.runName'
MLFLOW_PARENT_RUN_ID = 'cral.tracking.parentRunId'
MLFLOW_USER = 'cral.tracking.user'
MLFLOW_SOURCE_TYPE = 'cral.tracking.source.type'
MLFLOW_SOURCE_NAME = 'cral.tracking.source.name'
MLFLOW_GIT_COMMIT = 'cral.tracking.source.git.commit'
MLFLOW_GIT_BRANCH = 'cral.tracking.source.git.branch'
MLFLOW_GIT_REPO_URL = 'cral.tracking.source.git.repoURL'
MLFLOW_LOGGED_MODELS = 'cral.tracking.models'
MLFLOW_PROJECT_ENV = 'cral.tracking.project.env'
MLFLOW_PROJECT_ENTRY_POINT = 'cral.tracking.project.entryPoint'
MLFLOW_DOCKER_IMAGE_URI = 'cral.tracking.docker.image.uri'
MLFLOW_DOCKER_IMAGE_ID = 'cral.tracking.docker.image.id'

MLFLOW_DATABRICKS_NOTEBOOK_ID = 'cral.tracking.databricks.notebookID'
MLFLOW_DATABRICKS_NOTEBOOK_PATH = 'cral.tracking.databricks.notebookPath'
MLFLOW_DATABRICKS_WEBAPP_URL = 'cral.tracking.databricks.webappURL'
MLFLOW_DATABRICKS_RUN_URL = 'cral.tracking.databricks.runURL'
# The SHELL_JOB_ID and SHELL_JOB_RUN_ID tags are used for tracking the
# Databricks Job ID and Databricks Job Run ID associated with an MLflow
# Project run
MLFLOW_DATABRICKS_SHELL_JOB_ID = 'cral.tracking.databricks.shellJobID'
MLFLOW_DATABRICKS_SHELL_JOB_RUN_ID = 'cral.tracking.databricks.shellJobRunID'
# The JOB_ID, JOB_RUN_ID, and JOB_TYPE tags are used for automatically
# recording Job information
# when MLflow Tracking APIs are used within a Databricks Job
MLFLOW_DATABRICKS_JOB_ID = 'cral.tracking.databricks.jobID'
MLFLOW_DATABRICKS_JOB_RUN_ID = 'cral.tracking.databricks.jobRunID'
MLFLOW_DATABRICKS_JOB_TYPE = 'cral.tracking.databricks.jobType'

MLFLOW_PROJECT_BACKEND = 'cral.tracking.project.backend'

# The following legacy tags are deprecated and will be removed by MLflow 1.0.
LEGACY_MLFLOW_GIT_BRANCH_NAME = 'cral.tracking.gitBranchName'
LEGACY_MLFLOW_GIT_REPO_URL = 'cral.tracking.gitRepoURL'

# Segmind Restricted Tags
MLFLOW_SEGMIND_RUN_NAME = 'segmind_run_name'
MLFLOW_SEGMIND_RUN_ALGO_NAME = 'segmind_run_algo_name'
MLFLOW_SEGMIND_USER_USERNAME = 'segmind_user_username'
MLFLOW_SEGMIND_USER_EMAIL = 'segmind_user_email'
