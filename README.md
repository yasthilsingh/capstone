# Capstone Project

This repository contains our capstone project code for analyzing NHANES data.

## Project structure

- `notebooks/`: exploratory analysis and modeling notebooks
- `src/`: reusable Python functions and scripts
- `scripts/`: utility scripts for checking project setup and data loading
- `data/`: local raw data; data files are not committed to GitHub
- `outputs/`: generated figures, tables, and model outputs
- `requirements.txt`: Python package dependencies

The `.gitkeep` files allow Git to preserve otherwise empty folders. They do not contain project data.

## Local setup

Clone the repository and enter the project folder:

```bash
git clone https://github.com/yasthilsingh/capstone.git
cd capstone
```

Create a virtual environment and install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
## Local data setup

The data used is provided in the "data" folder in this repository. It is NHANES data initially located at https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2017-2020

The folder structure is:

```text
data/
├── Demographic/
├── Examination/
├── Lab/
└── Questionnaire/
```

The current data loader uses the `Demographic`, `Examination`, `Lab`, and `Questionnaire` folders. 

Raw data and generated outputs are ignored by Git and should not be committed.

## NHANES data loader

The reusable NHANES loader is located here:

```text
src/nhanes_loader.py
```

The loader reads selected NHANES `.xpt` files from the local `data/` folder and merges them into one master dataframe using `SEQN`, the NHANES participant identifier.

To check that the local data setup works, run:

```bash
python -m scripts.check_data_load
```

When tested locally, the loader successfully read 21 NHANES files and produced a merged master dataframe with 15,560 rows and 344 columns.

## Collaboration workflow

Please use a separate branch for your work instead of committing directly to `main`.

1. Update your local `main` branch:

   ```bash
   git switch main       # Move to your local main branch
   git pull origin main  # Download and apply the latest changes from GitHub's main branch
   ```

   Do this before starting new work so your branch begins from the latest version.

2. Create a branch for your task:

   ```bash
   git switch -c feature/your-task-name  # Create a new task branch and switch to it
   ```

   - `-c` creates a new branch.
   - Git also switches you onto the new branch.
   - Replace `your-task-name` with a short description of your task, such as `sleep-analysis`.

3. Commit your changes:

   ```bash
   git status                              # Review changed files before staging them
   git add .                               # Stage all changes in the project for the next commit
   git commit -m "Describe your change"    # Save the staged changes as a named snapshot
   ```

4. Push your branch:

   ```bash
   git push -u origin feature/your-task-name  # Upload the branch and connect it to GitHub
   ```

   - `git push` uploads your commits.
   - `origin` refers to the GitHub repository.
   - `feature/your-task-name` is the branch being uploaded.
   - `-u` connects your local branch to its GitHub counterpart.

   After the first push, future uploads from that branch only require:

   ```bash
   git push
   ```

5. Open a pull request on GitHub to merge your branch into `main`.

   For example, open a pull request from `feature/sleep-analysis` into `main`. Teammates can review it before it is merged.

Overall flow:

**Update main → Create branch → Make changes → Add → Commit → Push → Pull request → Merge**
