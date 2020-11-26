![PyPI - Python Version](https://img.shields.io/pypi/pyversions/segmind?style=plastic)
![GitHub Workflow Status (event)](https://img.shields.io/github/workflow/status/segmind/segmind/CI?event=push)
![GitHub Workflow Status (event)](https://img.shields.io/github/workflow/status/segmind/segmind/Upload%20Python%20Package?event=publish)
![PyPI - Downloads](https://img.shields.io/pypi/dm/segmind)
![PyPI](https://img.shields.io/pypi/v/segmind)
![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/segmind/segmind)
## Installation

`pip3 install -r requirements.txt`

## Code style

### Python
We adopt [PEP8](https://www.python.org/dev/peps/pep-0008/) as the preferred code style.

We use the following tools for linting and formatting:
- [flake8](http://flake8.pycqa.org/en/latest/): linter
- [yapf](https://github.com/google/yapf): formatter
- [isort](https://github.com/timothycrosley/isort): sort imports

Style configurations of yapf and isort can be found in [setup.cfg](../setup.cfg).

We use [pre-commit hook](https://pre-commit.com/) that checks and formats for `flake8`, `yapf`, `isort`, `trailing whitespaces`,
 fixes `end-of-files`, sorts `requirments.txt` automatically on every commit.
The config for a pre-commit hook is stored in [.pre-commit-config](../.pre-commit-config.yaml).

After you clone the repository, you will need to install initialize pre-commit hook.

```
pip install -U pre-commit
```

From the repository folder
```
pre-commit install
```

After this on every commit check code linters and formatter will be enforced.
