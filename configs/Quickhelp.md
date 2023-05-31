**Manage git repo**

git status - to check if some files have been updated or created newly and need to be added to local repo

git add . - add them all files or a specific file to git tracking

git commit -m "[message describing what you did]" to save a current state of repo as a snapshot

git pull - if one of us has made any changes on the remote repo and we would like to pull these changes

git push -u origin main - push changes to remote repo into origin/main ie the remote main branch or any other feature branch

git branch -a - list branches in repo

git branch --delete namebranch

git remote prune origin /  git remote prune origin --dry-run (check before getting rid of stale branches)

https://medium.com/@steveamaza/how-to-write-a-proper-git-commit-message-e028865e5791

https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github

https://education.github.com/git-cheat-sheet-education.pdf

https://www.earthdatascience.org/workshops/intro-version-control-git/

**Check pytorch and cuda version is working:**

import torch
torch.__version__
torch.cuda.is_available()

**Export env from conda**
conda env export > tsuML.yml
conda activate /mnt/beegfs/nragu/tsunami/env

in IUSS HPC my conda env: /mnt/beegfs/nragu/tsunami/env

**Utils**
conda install -c conda-forge gdown
gdown "fileid"

**Environmental Variables in /home/${USER}/.bash_profile**
IUSSHPC:
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
export MLDir="/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami"
export SimDir="/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/data/simu/PS_manning003"

CARISMA(was hardcoded in the code):
export MLDir="/mnt/data/nragu/Tsunami/INGV/IUSS_INGV_Repo"
export SimDir="/mnt/data/nragu/Tsunami/INGV/IUSS_INGV_Repo/data/simu/PS_manning003"

INGV:

**System Arguments as variables in sbatch scripts**
sbatch run.sbatch CT 1212
sbatch run.sbatch SR 1212

**Philosophy**
run sample and then check before launching full dataset
