import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
LOAN_FILE = os.path.join(DATA_DIR, 'Loan_status_2007-2020Q3.gzip')
OBJECT_DIR = os.path.join(PROJECT_ROOT, 'object')
PKL_DIR = os.path.join(OBJECT_DIR, 'pkl')
MODEL_DIR = os.path.join(OBJECT_DIR, 'model')