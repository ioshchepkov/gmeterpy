.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

Development Setup
-----------------

gMeterPy requires Python 3.12 or newer. Create a virtual environment and
install the project in editable mode with its development tools:

.. code:: bash

    python -m venv .venv
    .venv/bin/python -m pip install -e ".[dev]"
    .venv/bin/pre-commit install

Before submitting a change, run the same checks used by CI:

.. code:: bash

    .venv/bin/ruff check .
    .venv/bin/ruff format --check .
    .venv/bin/pytest
    .venv/bin/python -m build
    .venv/bin/twine check --strict dist/*
    .venv/bin/check-wheel-contents dist/*.whl

Use ``ruff check --fix .`` and ``ruff format .`` to apply automatic fixes.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/opengrav/gmeterpy/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

gMeterPy could always use more documentation, whether as part of the
official gMeterPy docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/opengrav/gmeterpy/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)
