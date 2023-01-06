Extending FLamby
----------------

FLamby is a living project and contributions by the FL community are
welcome.

If you would like to add another cross-silo dataset **with natural
splits**, please fork the repository and do a Pull-Request following the
guidelines described below.

Similarly, you can propose pull requests introducing novel training
algorithms or models.

Contribution Guidelines
-----------------------

After installing the package in dev mode
(``pip install -e .[all_extra]``) You should also initialize
``pre-commit`` by running:

::

    pre-commit install

The ``pre-commit`` tool will automatically run
`black <https://github.com/psf/black>`__ and
`isort <https://github.com/PyCQA/isort>`__ and check
`flake8 <https://flake8.pycqa.org/en/latest/>`__ compatibility. Which
will format the code automatically making the code more homogeneous and
helping catching typos and errors.

Looking and or commenting the open issues is a good way to start. Once
you have found a way to contribute the next steps are:

-  Following the installation instructions (with -e and all_extra)
-  Installing pre-commit by running: ``pre-commit install`` at the root of the repository
-  Creating a new branch following the convention name\_contributor/short\_explicit\_name-wpi:
   ``git checkout -b name_contributor/short_explicit_name-wpi``
-  Potentially pushing the branch to origin with :
   ``git push origin name_contributor/short_explicit_name-wpi`` 
-  Working on the branch locally by making commits frequently:
   ``git commit -m "explicit description of the commit's content"`` 
-  Once the branch is ready or after considering you have made significant
   progresses opening a Pull Request using Github interface, selecting your
   branch as a source and the target to be the main branch and creating the
   PR **in draft mode** after having made **a detailed description of the
   content of the PR** and potentially linking to related issues. 
-  Rebasing the branch onto main by doing ``git fetch origin`` and
   ``git rebase origin/main``, solving potential conflicts adding the
   resolved files ``git add myfile.py`` then continuing with
   ``git rebase --continue`` until the rebase is complete. Then pushing the
   branch to origin with ``git push origin --force-with-lease``. 
-  Waiting for reviews then commiting and pushing changes to comply with the reviewer's requests 
-  Once the PR is approved click on the arrow on the right of the merge button to select rebase and click on it
