import pandas as pd

path = r"C:\Users\Tanish_Gupta\ai-phishing-detector\dataset\spam_ham_dataset.csv"

def load_data(file_path):
    return pd.read_csv(file_path)


