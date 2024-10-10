import pandas as pd

def calculate_variance(file_path):
    # Read the CSV file
    data = pd.read_csv(file_path)
    
    # Calculate the variance for each column
    variances = data.var()
    
    return variances

def main():
    # File paths
    imu_file = './Data/0-steady-state_accel.csv'
    fts_file = './Data/0-steady-state_wrench.csv'
    
    # Calculate variances
    imu_variances = calculate_variance(imu_file)
    fts_variances = calculate_variance(fts_file)
    
    # Print the results
    print("IMU Signal Variances:")
    print(imu_variances)
    print("\nFTS Signal Variances:")
    print(fts_variances)

main()

