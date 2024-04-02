from DataHandler import DataAccess
from ChartReport import Chart
from LinearRegression import Linear_regression

def main():
    train_data_csv = "writtenAssignment/train.csv"
    ideal_data_csv = "writtenAssignment/ideal.csv"
    test_data_csv = "writtenAssignment/test.csv"
    db_path = "myDataBase.db"
    ideal_chart_save_path = "writtenAssignment/charts/ideal/"
    train_chart_save_path = "writtenAssignment/charts/train/"
    mapping_chart_save_path = "writtenAssignment/charts/mapping/"
    max_dev_save_chart = "writtenAssignment/charts/max_deviation/"

    data_access = DataAccess(db_path)
    data_access.load_csv_to_db(train_data_csv, ideal_data_csv, test_data_csv, db_path)

    chart_report = Chart(data_access)

    linear_regr = Linear_regression(data_access)
    train_data_line_eq_dict = linear_regr.line_equation_generator(data_access.get_train_data_df())
    least_square_dict = linear_regr.least_squares_method(data_access.get_ideal_data_df(), train_data_line_eq_dict)
    best_fit_line_dict = linear_regr.best_fit_line_ideal_func(least_square_dict)
    chart_report.generate_bestfit_chart(best_fit_line_dict, ideal_chart_save_path, train_chart_save_path)
    chart_report.generate_least_square_barchart(least_square_dict, ideal_chart_save_path)

    max_deviation_train_ideal_dict = {}
    for train_col_y in train_data_line_eq_dict:
        ideal_col_y = (list(best_fit_line_dict[train_col_y].items())[0][0])
        max_deviation = linear_regr.max_deviation_calc(train_col_y, ideal_col_y, train_data_line_eq_dict[train_col_y], max_dev_save_chart)
        max_deviation_train_ideal_dict[train_col_y] = {ideal_col_y: max_deviation}

    print(max_deviation_train_ideal_dict)
    mapping_test_data_dict = linear_regr.validate_max_deviation_test_data(max_deviation_train_ideal_dict)
    
    data_access.create_mapping_test_data_table(mapping_test_data_dict)
    print(data_access.get_mapping_test_data_table())

    chart_report.generate_map_test_data_chart(mapping_test_data_dict, max_deviation_train_ideal_dict, mapping_chart_save_path)


if __name__ == "__main__":
    main()