#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

// Function to load CSV file
std::vector<std::vector<double>> load_csv(const std::string& filename) {
    std::ifstream file(filename);
    std::string line;
    std::vector<std::vector<double>> data;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;

        // Parse each line by comma
        while (std::getline(ss, value, ',')) {
            row.push_back(std::stod(value));
        }

        data.push_back(row);
    }

    return data;
}

// Function to plot columns from the CSV data
void plot_data(const std::vector<std::vector<double>>& data, const std::vector<int>& columns, const std::string& title) {
    std::vector<double> t = data[0];  // Assuming 't' is in the first column
    
    for (int col : columns) {
        std::vector<double> y;
        for (size_t i = 0; i < data.size(); ++i) {
            y.push_back(data[i][col]);
        }
        plt::plot(t, y, {{"label", "Column " + std::to_string(col)}});
    }

    plt::xlabel("Time (microseconds)");
    plt::ylabel("Value");
    plt::title(title);
    plt::legend();
    plt::grid(true);
    plt::show();
}

int main() {
    // Paths to your CSV files
    std::string file_path = "path/to/your/0-calibration_fts-accel.csv";

    // Load the CSV data
    std::vector<std::vector<double>> data = load_csv(file_path);

    // Columns to plot (e.g., ax, ay, az)
    std::vector<int> columns_to_plot = {1, 2, 3};  // Change the index according to the column structure

    // Plot the data
    plot_data(data, columns_to_plot, "Calibration Acceleration");

    return 0;
}
