import pandas as pd
import numpy as np
from scipy import stats
import math
# internal import
from ChartReport import Chart

class Linear_regression:
    """
    Linear Regression Class

    This class provides functionality for linear regression analysis, including calculating maximum deviation,
    generating best-fit lines, and validating test data against the ideal data.

    Attributes:
        db_access (DataAccess): An instance of the DataAccess class for accessing the database.
        ideal_data_df (DataFrame): DataFrame containing the ideal data from the database.
        train_data_df (DataFrame): DataFrame containing the training data from the database.
        test_data_df (DataFrame): DataFrame containing the test data from the database.
    """
    def __init__(self, db_access):
        """
        Initialize LinearRegression with a database access instance.

        Args:
            db_access (DataAccess): An instance of the DataAccess class.
        """
        self.db_access = db_access
        self.ideal_data_df = self.db_access.get_ideal_data_df()
        self.train_data_df = self.db_access.get_train_data_df()
        self.test_data_df = self.db_access.get_test_data_df()

    def max_deviation_calc(self, train_column_y, ideal_column_y, train_line_eq, max_dev_save_chart):
        """
        Calculate the maximum deviation between train and ideal data and generate a chart.

        Args:
            train_column_y (str): Column name of the train data representing the dependent variable.
            ideal_column_y (str): Column name of the ideal data representing the dependent variable.
            train_line_eq (dict): Dictionary containing the slope and intercept of the best-fit line for train data.
            max_dev_save_chart (str): File path to save the generated chart.

        Returns:
            float: Maximum deviation between train and ideal data.
        """
        self.max_deviation_df = pd.DataFrame()
        self.max_deviation_df['x'] = self.train_data_df['x']
        self.max_deviation_df[train_column_y] = self.train_data_df[train_column_y]
        self.max_deviation_df[ideal_column_y] = self.ideal_data_df[ideal_column_y]
        self.max_deviation_df['y(bestfit)'] = (train_line_eq['slope'] * self.train_data_df['x']) + train_line_eq['intercept']
        self.max_deviation_df['deviation'] = round(abs(self.ideal_data_df[ideal_column_y] - self.max_deviation_df['y(bestfit)']), 4)
        max_deviation = self.max_deviation_df['deviation'].max()
        max_deviation = round(max_deviation, 4)
        Chart.generate_max_deviation_graph(self, self.max_deviation_df, max_deviation, train_column_y, ideal_column_y, max_dev_save_chart)
        return max_deviation
    
    def is_within_max_deviation(self, test_data, train_data, ideal_data, max_deviation):
        """
        Check if the test data is within the maximum deviation of train and ideal data.

        Args:
            test_data (float): Test data value.
            train_data (float): Train data value.
            ideal_data (float): Ideal data value.
            max_deviation (float): Maximum deviation threshold.

        Returns:
            bool: True if test data is within max deviation, False otherwise.
        """
        # Calculate the absolute difference between test data and the combination of train and ideal data
        diff_train = np.abs(test_data - train_data)
        diff_ideal = np.abs(test_data - ideal_data)
        
        # Check if the maximum deviation is less than or equal to the predefined max_deviation
        return (diff_train <= max_deviation).any() or (diff_ideal <= max_deviation).any()

    def line_equation_generator(self, train_data_df):
        """
        Generate line equations for each dependent variable in the train data.

        Args:
            train_data_df (DataFrame): DataFrame containing the train data.

        Returns:
            dict: Dictionary containing line equations for each dependent variable.
        """
        train_data_line_equation_dict = {}
        for col_name in train_data_df.columns[:]:
            if col_name.find('y') != -1:
                slope_intercept_dict = {}
                dependant_dataset = train_data_df[col_name]
                independent_dataset = train_data_df['x']
                slope, intercept, r_value, p_value, std_err = stats.linregress(independent_dataset, dependant_dataset)
                slope_intercept_dict["intercept"] = intercept
                slope_intercept_dict["slope"] = slope
                slope_intercept_dict["r_value"] = r_value
                slope_intercept_dict["p_value"] = p_value
                slope_intercept_dict["std_err"] = std_err
                train_data_line_equation_dict[col_name] = slope_intercept_dict
        return train_data_line_equation_dict

    def least_squares_method(self, ideal_data_df, line_equ_train_data_dict):
        """
        Apply the least squares method to calculate root mean square error for each dependent variable.

        Args:
            ideal_data_df (DataFrame): DataFrame containing the ideal data.
            line_equ_train_data_dict (dict): Dictionary containing line equations for train data.

        Returns:
            dict: Dictionary containing root mean square error for each dependent variable.
        """
        least_square_dict = {}
        for line_eq_train_data_idx in line_equ_train_data_dict:
            intercept = line_equ_train_data_dict[line_eq_train_data_idx]['intercept']
            slope = line_equ_train_data_dict[line_eq_train_data_idx]['slope']
            ideal_least_square_dict = {}
            for col_name in ideal_data_df.columns[:]:
                least_square_df = pd.DataFrame()
                least_square_df['x'] = ideal_data_df['x']
                if col_name.find('y') != -1:
                    least_square = 0
                    least_square_df['y_actual'] = ideal_data_df[col_name]
                    least_square_df['y_predicted'] = (least_square_df['x'] * slope) + intercept
                    least_square_df['Residual_err'] = least_square_df['y_actual'] - least_square_df['y_predicted']
                    least_square_df['Residual_err_square'] = least_square_df['Residual_err'] * least_square_df['Residual_err']
                    least_square = least_square_df['Residual_err_square'].sum()
                    Mean_least_square = least_square / len(least_square_df['y_actual'])
                    Root_mean_square_error = math.sqrt(Mean_least_square)
                    ideal_least_square_dict[col_name] = Root_mean_square_error
                    least_square_dict[line_eq_train_data_idx] = ideal_least_square_dict
        return least_square_dict

    def best_fit_line_ideal_func(self, least_square_dict):
        """
        Determine the best-fit line for each dependent variable based on the least square errors.

        Args:
            least_square_dict (dict): Dictionary containing root mean square errors for each dependent variable.

        Returns:
            dict: Dictionary containing the best-fit line for each dependent variable.
        """
        best_fit_line = {}
        for train_line in least_square_dict:
            least_square = min(least_square_dict[train_line].values())
            least_square_key_value = {key: value 
                                      for key, value in least_square_dict[train_line].items() 
                                      if value == least_square}
            best_fit_line[train_line] = least_square_key_value
        return best_fit_line

    def validate_max_deviation_test_data(self, max_deviation_train_ideal_dict):
        """
        Validate test data against the maximum deviation of train and ideal data.

        Args:
            max_deviation_train_ideal_dict (dict): Dictionary containing maximum deviation for each combination of train and ideal data.

        Returns:
            dict: Dictionary containing mapping test data points for each maximum deviation and ideal column combination.
        """
        mapping_test_data_dict = {}
        for max_deviation_idx in max_deviation_train_ideal_dict:
            for ideal_col_y_idx in(max_deviation_train_ideal_dict[max_deviation_idx]):
                mapping_data_set_dict = {}
                ideal_col_y = ideal_col_y_idx
                max_deviation = max_deviation_train_ideal_dict[max_deviation_idx][ideal_col_y]
                max_deviation_mapper_df = pd.DataFrame()
                ideal_x = self.ideal_data_df['x'].to_numpy()
                ideal_y = self.ideal_data_df[ideal_col_y].to_numpy()
                ideal_slope, ideal_intercept, r_value, p_value, std_err = stats.linregress(ideal_x, ideal_y)
                ideal_intercept = round(ideal_intercept, 4)
                ideal_slope = round(ideal_slope, 4)
                max_deviation_mapper_df['x'] = self.ideal_data_df['x']
                max_deviation_mapper_df[ideal_col_y] = self.ideal_data_df[ideal_col_y]
                max_deviation_mapper_df['y_bestfit'] = (ideal_slope * self.ideal_data_df['x']) + ideal_intercept
                max_deviation_mapper_df['y_upperband'] = max_deviation_mapper_df['y_bestfit'] + max_deviation
                max_deviation_mapper_df['y_lowerband'] = max_deviation_mapper_df['y_bestfit'] - max_deviation
                max_deviation_mapper_df.set_index('x', inplace=True)
                for index, test_data_row in self.test_data_df.iterrows():
                    x_index = test_data_row['x']
                    y_value = test_data_row['y']
                    ideal_data = (max_deviation_mapper_df.loc[x_index])
                    if (y_value >= ideal_data['y_lowerband'] and y_value <= ideal_data['y_upperband']):
                        mapping_data_point_dict = {}
                        mapping_data_point_dict["x"] = x_index
                        mapping_data_point_dict["y"] = y_value
                        mapping_data_point_dict["ideal_column"] = ideal_col_y_idx
                        mapping_data_point_dict["y_upperband"] = ideal_data['y_upperband']
                        mapping_data_point_dict["y_lowerband"] = ideal_data['y_lowerband']
                        mapping_data_set_dict[x_index] = mapping_data_point_dict 
                        mapping_test_data_dict[max_deviation_idx + "_" + ideal_col_y_idx] = mapping_data_set_dict       
        return mapping_test_data_dict