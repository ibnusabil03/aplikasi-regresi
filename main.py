import os

def test_regression():
    # Path ke file CSV
    file_path = 'Student_Performance.csv'

    # Uji regresi linear dan eksponensial
    print("Testing Regressions:")
    os.system(f'python regression_models.py {file_path}')

if __name__ == "__main__":
    test_regression()
