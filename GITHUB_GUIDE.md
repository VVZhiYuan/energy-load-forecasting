# Publish This Project To GitHub

The raw UCI dataset is 711 MB and must not be uploaded to GitHub. The project's
`.gitignore` already excludes `data/raw/` and `data/processed/`, while keeping
the code, notebooks, generated charts, and result tables.

## 1. Create An Empty Repository On GitHub

1. Sign in to https://github.com/.
2. Select **New repository**.
3. Name it `energy-load-forecasting`.
4. Choose **Public** for a job-search portfolio.
5. Do not add a README, `.gitignore`, or license on GitHub because the local
   project already contains them.
6. Select **Create repository**.

## 2. Initialize The Local Repository

Open the VS Code terminal in this project folder and run:

```powershell
git init
git add .
git status
```

Before committing, confirm that `data/raw/LD2011_2014.txt` is not listed by
`git status`. The PNG files under `reports/figures/` and the CSV files under
`reports/tables/` should be listed.

Configure the author identity shown on your commits. Use your GitHub display
name and the email attached to your GitHub account:

```powershell
git config --global user.name "YOUR_NAME"
git config --global user.email "YOUR_GITHUB_EMAIL"
```

Then create the first commit:

```powershell
git commit -m "Build initial electricity load forecasting EDA"
git branch -M main
```

## 3. Connect And Push

Copy the repository URL shown by GitHub, replace `<YOUR_USERNAME>`, and run:

```powershell
git remote add origin https://github.com/<YOUR_USERNAME>/energy-load-forecasting.git
git push -u origin main
```

GitHub may open a browser sign-in window the first time. Complete that sign-in,
then return to VS Code.

## 4. Push Later Updates

After changing code, notebooks, figures, or documentation:

```powershell
git status
git add .
git commit -m "Describe the completed change"
git push
```

## 5. Portfolio Checklist

- Keep the repository public.
- Add the repository URL to your CV and LinkedIn project section.
- Keep the generated figures visible in the README.
- Cite the UCI dataset and its DOI.
- Never commit `.venv/`, raw datasets, API keys, passwords, or tokens.
- Use meaningful commits for baselines, machine learning models, robustness
  experiments, and the final dashboard.
