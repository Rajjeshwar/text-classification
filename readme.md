folder structure:
1. Dataframe: contains the processed training dataframes for all models
2. Notebooks: contains the .ipynb code execution for all models and explainability
3. Notebooks/Models: contains trained models
4. script: Contains all .py files for training models
(We are sharing the dataframes and trained models seperately in drive link, please put them in the folder structure for execution)

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
