
Folder structure:
1. Notebooks: contains the .ipynb code execution for all models and explainability
3. Notebooks/Models: contains trained models
4. Script: Contains all .py files for training models
(We are sharing the dataframes and trained models seperately in drive link, please put them in the folder structure for execution)

**Refer to _Text_Classification_Challenge_report.pdf_ for details on the process followed and the results from GradCAM explainability outputs.**

Installation:
create a python virtual enviornment from given requirements.txt file

`
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
`

Execution:
For training a given model, execute the corresponding .py file. e.g., for training LSTM ensemble simply run

`
python3 lstm.py
`

NOTE: due to a system error some of the files were lost during the initial push, please refer to the report for explainability importance map examples.
