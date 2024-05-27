import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def linear_regression(X, y):
    # Model Linear
    linear_model = LinearRegression()
    linear_model.fit(X, y)

    # Prediksi
    y_pred_linear = linear_model.predict(X)

    # Plot hasil regresi linear
    plt.scatter(X['Hours Studied'], y, color='blue', label='Data Asli')
    plt.plot(X['Hours Studied'], y_pred_linear, color='red', label='Regresi Linear')
    plt.xlabel('Hours Studied')
    plt.ylabel('Performance Index')
    plt.legend()
    plt.title('Regresi Linear')
    plt.show()

    # Menghitung galat RMS
    rms_linear = np.sqrt(mean_squared_error(y, y_pred_linear))
    print(f'Galat RMS (Regresi Linear): {rms_linear}')

def exponential_regression(X, y):
    # Mengubah data untuk model eksponensial
    X_exp = np.log1p(X)
    y_exp = np.log1p(y)

    # Model Eksponensial
    exp_model = LinearRegression()
    exp_model.fit(X_exp, y_exp)

    # Prediksi
    y_pred_exp = exp_model.predict(X_exp)
    y_pred_exp = np.expm1(y_pred_exp)

    # Plot hasil regresi eksponensial
    plt.scatter(X['Hours Studied'], y, color='blue', label='Data Asli')
    plt.plot(X['Hours Studied'], y_pred_exp, color='green', label='Regresi Eksponensial')
    plt.xlabel('Hours Studied')
    plt.ylabel('Performance Index')
    plt.legend()
    plt.title('Regresi Eksponensial')
    plt.show()

    # Menghitung galat RMS
    rms_exp = np.sqrt(mean_squared_error(y, y_pred_exp))
    print(f'Galat RMS (Regresi Eksponensial): {rms_exp}')

if __name__ == "__main__":
    import sys
    file_path = sys.argv[1] if len(sys.argv) > 1 else 'Student_Performance.csv'
    
    # Membaca data dari file CSV
    data = pd.read_csv(file_path)

    # Memilih kolom yang relevan
    X = data[['Hours Studied', 'Sample Question Papers Practiced']]
    y = data['Performance Index']

    # Menjalankan regresi linear
    print("Running Linear Regression:")
    linear_regression(X, y)

    # Menjalankan regresi eksponensial
    print("Running Exponential Regression:")
    exponential_regression(X, y)
