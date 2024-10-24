#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "fusion.h"

std::vector<Eigen::VectorXd> loadCSV(const std::string &filename, int cols) {
    std::ifstream file(filename);
    std::vector<Eigen::VectorXd> data;
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        Eigen::VectorXd row(cols);
        for (int i = 0; i < cols; ++i) {
            ss >> row(i);
            if (ss.peek() == ',') ss.ignore();
        }
        data.push_back(row);
    }
    return data;
}

int main() {
    Fusion fusion;
    
    // Load the IMU and FTS data from CSV files (replace with actual filenames)
    std::vector<Eigen::VectorXd> imu_data = loadCSV("./Data/1-baseline_accel.csv", 3);  // 3 columns for acceleration (x, y, z)
    std::vector<Eigen::VectorXd> fts_data = loadCSV("./Data/1-baseline_wrench.csv", 6); // 6 columns for force and torque (fx, fy, fz, tx, ty, tz)

    std::vector<Eigen::VectorXd> estimated_wrenches;

    // Assume both datasets have the same length (for simplicity)
    for (size_t i = 0; i < imu_data.size(); ++i) {
        Eigen::VectorXd wrench = fusion.estimateContactWrench(imu_data[i], fts_data[i]);
        estimated_wrenches.push_back(wrench);

        // Print estimated contact wrench (forces and torques)
        std::cout << "Estimated Wrench " << i << ": " << wrench.transpose() << std::endl;
    }

    return 0;
}
