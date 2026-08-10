# Assignment 1 Part A Postgraduate Starter

For ISYS3476: Managing Semi-structured and Unstructured Data.

## Setup

The VS Code steps below provide a guided setup. If you are comfortable with Python environments or the terminal, use your preferred setup and skip to the self-checker. In either case, use Python 3.10 to 3.14, install `requirements.txt` and the required NLTK resources, and follow the permitted-import list in the specification.

For guided setup in VS Code:

1. Download the starter ZIP from Canvas.
2. Extract the ZIP first. On Windows, right-click it and select **Extract All**. On macOS, double-click it. Do not work inside the ZIP.
3. Open VS Code, select **File > Open Folder**, and choose the extracted `ir-ass1-partA-startcode` folder itself.
4. If VS Code displays **Restricted Mode**, select **Manage** and trust the extracted course folder. Confirm that `README.md`, `requirements.txt`, and `utils/` are visible in the VS Code file list.
5. Press **Ctrl+Shift+P** on Windows or **Cmd+Shift+P** on macOS. Type `Python: Create Environment` and select it from the results. On the next screen, do not select **Quick Create**. Select **venv**, the second option. If the command does not appear, install or update the Microsoft Python and Python Environments extensions in VS Code.
6. Select an installed Python version from 3.10 to 3.14. If no supported version is listed, install one from [python.org](https://www.python.org/downloads/) and reopen VS Code.
7. Select **Install project dependencies**. On the next screen, keep `requirements.txt` selected and click **OK**. VS Code will create a local `.venv`, install the packages, and select the new environment for this folder.
8. When environment creation finishes, select **Terminal > New Terminal**. The terminal prompt should begin with `(.venv)`, which confirms that the new environment is active. Then run:

```bash
python -m nltk.downloader punkt punkt_tab
```

## Self-Checker

Run the checker from the same VS Code terminal in the extracted starter folder.

```bash
python -m utils.check_submission_partA
```

The checker will report task failures before you complete the implementation. This is expected. For Task 1 it reports Profile A, Profile B, and the combined dev Average Hit@K. Before submitting, resolve every `FAIL`. Passing confirms basic runnability and contract compliance, but it does not guarantee full correctness or full marks.

## Submission

Submit one ZIP of your completed assignment on Canvas. Keep the folder structure unchanged and include all required files, including `AGENTS.md` and `DEVELOPMENT.md`.

Name the submitted ZIP as `YourName_s123456.zip` (e.g., `ZhuangLi_s123456.zip`).

Only the ZIP submitted on Canvas is marked.

Do not include virtual environments such as `.venv/`, `env/`, `venv/`, or `.conda/`.

## AI Assistance Policy

AI tools must not be used to generate a full solution for this assessment.

## Engineering Development Evidence

Complete `DEVELOPMENT.md` with the two requested excerpts. For each, include the failed output, briefly state what you changed, and show the rerun result. This is engineering evidence, not design or experimental analysis.

## Important Contracts

- Do not change required function names, argument order, or return types.
- Keep your implementation inside the repository.
- Keep the provided data files in place.
- Inspect both `data/dev_profile_a` and `data/dev_profile_b` when developing Task 1.
