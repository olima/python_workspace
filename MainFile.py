import os
#internal import
from DataHandler import DataAccess
from ChartReport import Chart
from LinearRegression import Linear_regression

def create_directory(directory_path):
        """
        Create a directory if it doesn't exist.

        Parameters:
        - directory_path (str): The path of the directory to be created.

        Returns:
        - bool: True if the directory was created or already exists, False otherwise.
        """
        if not os.path.exists(directory_path):
            try:
                os.makedirs(directory_path)
                print(f"Directory '{directory_path}' created successfully.")
                return True
            except OSError as e:
                print(f"Error: Failed to create directory '{directory_path}': {e}")
                return False
        else:
            print(f"Directory '{directory_path}' already exists.")
            return True

def main():
    """
    Main function to execute the data handling, chart generation, and linear regression processes.

    This function performs the following steps:
    1. Define file paths and database path.
    2. Create an instance of DataAccess to handle database operations.
    3. Load CSV data into the database using DataAccess.
    4. Create an instance of Chart to generate charts.
    5. Create an instance of Linear_regression to perform linear regression operations.
    6. Generate best-fit line charts and least squares bar charts.
    7. Calculate maximum deviation between training and ideal data points.
    8. Create mapping of maximum deviation test data.
    9. Create a mapping test data table in the database.
    10. Generate a chart illustrating the mapping of test data.

    """
    train_data_csv = "train.csv"
    ideal_data_csv = "ideal.csv"
    test_data_csv = "test.csv"
    db_path = "myDataBase.db"
    ideal_chart_save_path = "charts/ideal/"
    create_directory(ideal_chart_save_path)
    train_chart_save_path = "charts/train/"
    create_directory(train_chart_save_path)
    mapping_chart_save_path = "charts/mapping/"
    create_directory(mapping_chart_save_path)
    max_dev_save_chart = "charts/max_deviation/"
    create_directory(max_dev_save_chart)

    # Step 2: Create an instance of DataAccess to handle database operations
    data_access = DataAccess(db_path)

    # Step 3: Load CSV data into the database using DataAccess
    data_access.load_csv_to_db(train_data_csv, ideal_data_csv, test_data_csv, db_path)

    # Step 4: Create an instance of Chart to generate charts
    chart_report = Chart(data_access)

    # Step 5: Create an instance of Linear_regression to perform linear regression operations
    linear_regr = Linear_regression(data_access)

    # Step 6: Generate best-fit line charts and least squares bar charts
    train_data_line_eq_dict = linear_regr.line_equation_generator(data_access.get_train_data_df())
    least_square_dict = linear_regr.least_squares_method(data_access.get_ideal_data_df(), train_data_line_eq_dict)
    best_fit_line_dict = linear_regr.best_fit_line_ideal_func(least_square_dict)
    chart_report.generate_bestfit_chart(best_fit_line_dict, ideal_chart_save_path, train_chart_save_path)
    chart_report.generate_least_square_barchart(least_square_dict, ideal_chart_save_path)

    # Step 7: Calculate maximum deviation between training and ideal data points
    max_deviation_train_ideal_dict = {}
    for train_col_y in train_data_line_eq_dict:
        ideal_col_y = (list(best_fit_line_dict[train_col_y].items())[0][0])
        max_deviation = linear_regr.max_deviation_calc(train_col_y, ideal_col_y, train_data_line_eq_dict[train_col_y], max_dev_save_chart)
        max_deviation_train_ideal_dict[train_col_y] = {ideal_col_y: max_deviation}

    print(max_deviation_train_ideal_dict)

    # Step 8: Create mapping of maximum deviation test data
    mapping_test_data_dict = linear_regr.validate_max_deviation_test_data(max_deviation_train_ideal_dict)

    # Step 9: Create a mapping test data table in the database
    data_access.create_mapping_test_data_table(mapping_test_data_dict)
    print(data_access.get_mapping_test_data_table())

    # Step 10: Generate a chart illustrating the mapping of test data
    chart_report.generate_map_test_data_chart(mapping_test_data_dict, max_deviation_train_ideal_dict, mapping_chart_save_path)

if __name__ == "__main__":
    main()