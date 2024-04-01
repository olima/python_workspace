import pandas as pd
from bokeh.io import show
from bokeh.layouts import row
from bokeh.plotting import figure, output_file, save, show
from bokeh.models import Band, ColumnDataSource, FactorRange
from bokeh.layouts import gridplot, column
from bokeh.palettes import magma
from scipy import stats

"""
    Chart Class

    This class provides functionality for generating various types of charts using Bokeh.

    Attributes:
        db_access (DataAccess): DataAccess object for accessing database.
        ideal_data_df (DataFrame): DataFrame containing ideal data.
        train_data_df (DataFrame): DataFrame containing training data.
        test_data_df (DataFrame): DataFrame containing test data.
    """

class Chart:
    def __init__(self, db_access):
        """
        Initialize the Chart class with a DataAccess object.

        Args:
            db_access (DataAccess): DataAccess object for accessing database.
        """
        self.db_access = db_access
        self.ideal_data_df = self.db_access.get_ideal_data_df()
        self.train_data_df = self.db_access.get_train_data_df()
        self.test_data_df = self.db_access.get_test_data_df()

    def generate_bestfit_chart(self, best_fit_line_dict, ideal_chart_save_path, train_chart_save_path):
        """
        Generate a best-fit chart.

        Args:
            best_fit_line_dict (dict): Dictionary containing best-fit line data.
            ideal_chart_save_path (str): Path to save the ideal chart.
            train_chart_save_path (str): Path to save the train chart.
        """
        self.create_regression_dataframe(best_fit_line_dict, ideal_chart_save_path,train_chart_save_path)

    def create_regression_dataframe(self, best_fit_line_dict, ideal_chart_save_path, train_chart_save_path):
        """
        Create a regression DataFrame.

        Args:
            best_fit_line_dict (dict): Dictionary containing best-fit line data.
            ideal_chart_save_path (str): Path to save the ideal chart.
            train_chart_save_path (str): Path to save the train chart.
        """
        # Create empty lists to store train and ideal charts
        train_charts = []
        ideal_charts = []

        for train_data_idx in best_fit_line_dict:
            chart_info = {}
            chart_info[train_data_idx] = best_fit_line_dict[train_data_idx]
            regression_data_df = pd.DataFrame()
            train_data_y_data = train_data_idx
            ideal_data_y_data = (list(best_fit_line_dict[train_data_idx].items())[0][0])
            regression_data_df['x_train'] = self.train_data_df['x']
            regression_data_df['y_train'] = self.train_data_df[train_data_y_data]
            regression_data_df['x'] = self.ideal_data_df['x']
            regression_data_df['y'] = self.ideal_data_df[ideal_data_y_data]
            
            # Generate train and ideal line charts and append them to the respective lists
            train_chart = self.generate_train_line_chart(regression_data_df, chart_info)
            train_charts.append(train_chart)
            ideal_chart = self.generate_ideal_line_chart(regression_data_df, chart_info)
            ideal_charts.append(ideal_chart)

        # Combine all train charts into a single layout
        combined_train_chart = column(train_charts)
        # Save the combined train chart to a single HTML file
        train_chart_filename = train_chart_save_path + "train_charts.html"
        output_file(filename=train_chart_filename, title="Combined Train Data Charts")
        try:
            # Show the combined train chart
            show(combined_train_chart)
        except Exception as _error:
            print(f"Error: GRAPH GENERATED UNSUCCESSFUL: {_error}")

        # Combine all ideal charts into a single layout
        combined_ideal_chart = column(ideal_charts)
        # Save the combined ideal chart to a single HTML file
        ideal_chart_filename = ideal_chart_save_path + "ideal_charts.html"
        output_file(filename=ideal_chart_filename, title="Combined Ideal Data Charts")
        try:
            # Show the combined ideal chart
            show(combined_ideal_chart)
        except Exception as _error:
            print(f"Error: GRAPH GENERATED UNSUCCESSFUL: {_error}")

    def generate_train_line_chart(self, regression_data_df, chart_info):
        """
        Generate a line chart for training data.

        Args:
            regression_data_df (DataFrame): DataFrame containing regression data.
            chart_info (dict): Information about the chart.

        Returns:
            train_chart: The generated train chart.
        """
        # Extract x and y data from the DataFrame
        x_train = regression_data_df['x_train'].to_numpy()
        y_train = regression_data_df['y_train'].to_numpy()
        slope_train, intercept_train, r_value, p_value, std_err = stats.linregress(x_train, y_train)
        _r_value = (r_value)
        _p_value = (p_value)
        _r_square = (r_value) * (r_value)
        y_predicted_train = [slope_train * idx + intercept_train for idx in x_train]
        _train_slope = round(slope_train, 4)
        _train_intercept = round(intercept_train, 4)
        train_line_eq_str = f" y = {_train_slope}x + {_train_intercept}"

        for train_data_column_idx_key, ideal_least_square_info_value in chart_info.items():
            train_data_column_idx = train_data_column_idx_key

        train_p1 = figure(title=f"Train data function ({train_data_column_idx})", x_axis_label='x ', y_axis_label=f'y = {train_data_column_idx}')
        train_p1.line(x_train, y_predicted_train, legend_label=f"line eq : {train_line_eq_str}", line_color="red", line_width=2)
        train_p1.scatter(x_train, y_train, fill_color="red", size=2)
        train_p1.line(x_train, y_predicted_train, legend_label=f"r_value : {round(_r_value, 5)}", line_color="red",line_width=2)
        train_p1.scatter(x_train, y_train, fill_color="red", size=2)
        train_p1.line(x_train, y_predicted_train, legend_label=f"r_square : {round(_r_square, 5)}", line_color="red", line_width=2)
        train_p1.scatter(x_train, y_train, fill_color="red", size=2)
        train_p1.line(x_train, y_predicted_train, legend_label=f"p_value : {round(_p_value, 5)}", line_color="red", line_width=2)
        train_p1.scatter(x_train, y_train, fill_color="red", size=2)
        train_p1.line(x_train, y_predicted_train, legend_label=f"std_err : {round(std_err, 5)}", line_color="red", line_width=2)
        train_p1.scatter(x_train, y_train, fill_color="red", size=2)

        return train_p1

    def generate_least_square_barchart(self, least_square_dict, ideal_chart_save_path):
        """
        Generate a least square bar chart.

        Args:
            least_square_dict (dict): Dictionary containing least square data.
            ideal_chart_save_path (str): Path to save the chart.
        """
        # Initialize empty lists to store data for each chart
        plots = []
        titles = []

        for train_idx, ideal_least_square in least_square_dict.items():
            ideal_idx_list = []
            least_square_value_list = []
            color_list = []
            for ideal_idx, least_square_value in ideal_least_square.items():
                ideal_idx_list.append(f"{round((least_square_value), 4)}:{ideal_idx}")
                least_square_value_list.append(round((least_square_value), 4))
                color_list.append("blue")
                index = least_square_value_list.index(min(least_square_value_list))
                color_list[index] = "red"

            # Create a Bokeh figure for each chart
            TOOLS = "pan,box_zoom,reset,save,zoom_out,zoom_in"
            p = figure(y_range=FactorRange(factors=ideal_idx_list), max_width=500, height=500, title=f"least square for ideal function w.r.t train dataset : {train_idx}", toolbar_location=None, tools=TOOLS)
            p.hbar(y=ideal_idx_list, right=least_square_value_list, height=0.5, color=color_list) 

            # Add the figure and title to the lists
            plots.append(p)
            titles.append(f"least square for ideal function w.r.t train dataset : {train_idx}")

        # Arrange the charts in a grid layout
        grid = gridplot(plots, ncols=1, toolbar_location=None)

        # Save the grid layout to an HTML file
        file_header = f"least_square_ideal_dataset_vs_train_dataset"
        chart_filename = ideal_chart_save_path + file_header + ".html"
        output_file(filename=chart_filename, title=file_header)

        try:
            # Show the grid layout
            show(grid)
        except Exception as _error:
            print(f"Error: GRAPH GENERATED UNSUCCESSFUL: {_error}")

    def generate_ideal_line_chart(self, regression_data_df, chart_info):
        """
        Generate a line chart for ideal data.

        Args:
            regression_data_df (DataFrame): DataFrame containing regression data.
            chart_info (dict): Information about the chart.
        """

        line_eq_train_ideal_dict = self.generate_line_equation(chart_info)
        _line_eq_train_dict = line_eq_train_ideal_dict['ideal_function']
        _line_eq_ideal_dict = line_eq_train_ideal_dict['train_function']
        _train_slope = _line_eq_train_dict['slope']
        _train_intercept = _line_eq_train_dict['intercept']
        _ideal_slope = _line_eq_ideal_dict['slope']
        _ideal_intercept = _line_eq_ideal_dict['intercept']
        ideal_line_eq_str = f" y = {_ideal_slope}x + {_ideal_intercept}"
        train_line_eq_str = f" y = {_train_slope}x + {_train_intercept}"

        for train_data_column_idx_key, ideal_least_square_info_value in chart_info.items():
            train_data_column_idx = train_data_column_idx_key
            ideal_data_column_idx = list(ideal_least_square_info_value.keys())[0]
            ideal_least_square = list(ideal_least_square_info_value.values())[0]

            train_p1 = figure(title=f"Train data function ({chart_info})", x_axis_label='x', y_axis_label='y')
            ideal_p1 = figure(title=f"Ideal data function ({ideal_data_column_idx})", x_axis_label='x', y_axis_label=f'y = {ideal_data_column_idx}')
            ideal_p3 = figure(title=f"Train Vs Ideal data function ({train_data_column_idx} & {ideal_data_column_idx})", x_axis_label='x', y_axis_label=f'y = {train_data_column_idx} & {ideal_data_column_idx} ')

            x_train = regression_data_df['x_train'].to_numpy()
            y_train = regression_data_df['y_train'].to_numpy()
            slope_train, intercept_train, r_value, p_value, std_err = stats.linregress(x_train, y_train)
            y_predicted_train = [slope_train * idx + intercept_train for idx in x_train]

            x_ideal = regression_data_df['x'].to_numpy()
            y_ideal = regression_data_df['y'].to_numpy()
            slope_ideal, intercept_ideal, r_value, p_value, std_err = stats.linregress(x_ideal, y_ideal)
            y_predicted_ideal = [slope_ideal * idx + intercept_ideal for idx in x_ideal]

            # Add training data to the plots
            train_p1.line(x_train, y_predicted_train, legend_label=f"Train Line Eq: {train_line_eq_str}", line_color="red", line_width=2)
            train_p1.scatter(x_train, y_train, fill_color="red", size=2)

            ideal_p1.line(x_ideal, y_predicted_ideal, legend_label=f"Ideal Line Equation: y = {slope_ideal:.4f}x + {intercept_ideal:.4f}", line_color="green", line_width=2)
            ideal_p1.scatter(x_ideal, y_ideal, fill_color="green", size=2)

            ideal_p3.line(x_train, y_predicted_train, legend_label=f"Train Line Equation: y = {slope_train:.4f}x + {intercept_train:.4f}", line_color="red", line_width=2)
            ideal_p3.scatter(x_train, y_train, fill_color="red", size=2)
            ideal_p3.line(x_ideal, y_predicted_ideal, legend_label=f"Ideal Line Equation: y = {slope_ideal:.4f}x + {intercept_ideal:.4f}, Least Square: {round(ideal_least_square, 4)}", line_color="green", line_width=2)
            ideal_p3.scatter(x_ideal, y_ideal, fill_color="green", size=2)

            ideal_chart = row(train_p1, ideal_p1, ideal_p3)
            #ideal_charts.append(ideal_chart)

        return ideal_chart

    def generate_line_equation(self, chart_info):
        """
        Generate line equations.

        Args:
            chart_info (dict): Information about the chart.

        Returns:
            dict: Dictionary containing line equations.
        """
        line_eq_train_ideal_dict = {}
        line_eq_ideal_dict = {}
        line_eq_train_dict = {}
        for train_data_column_idx_key, ideal_least_square_info_value in chart_info.items():
            _train_data_column_y = train_data_column_idx_key
            for ideal_data_column_idx_key, ideal_least_square_value in ideal_least_square_info_value.items():
                _ideal_data_column_y = ideal_data_column_idx_key
                ideal_x = self.ideal_data_df['x'].to_numpy()
                ideal_y = self.ideal_data_df[_ideal_data_column_y].to_numpy()
                train_x = self.train_data_df['x'].to_numpy()
                train_y = self.train_data_df[_train_data_column_y].to_numpy() 

        ideal_slope, ideal_intercept, r_value, p_value, std_err = stats.linregress(ideal_x, ideal_y)
        train_slope, train_intercept, r_value, p_value, std_err = stats.linregress(train_x, train_y)
        line_eq_train_dict['slope'] = round(train_slope, 4)
        line_eq_train_dict['intercept'] = round(train_intercept, 4)
        line_eq_ideal_dict['slope'] = round(ideal_slope, 4)
        line_eq_ideal_dict['intercept'] = round(ideal_intercept, 4)
        line_eq_train_ideal_dict['ideal_function'] = line_eq_train_dict
        line_eq_train_ideal_dict['train_function'] = line_eq_ideal_dict
        return line_eq_train_ideal_dict
    
    def generate_mapper_graph(self, mapper_point_data, max_deviation_train_ideal_dict, mapping_chart_save_path):
        """
        Generate a mapper graph.

        Args:
            mapper_point_data (dict): Mapper point data.
            max_deviation_train_ideal_dict (dict): Maximum deviation data.
            mapping_chart_save_path (str): Path to save the chart.
        """
        df_in_range_point = pd.DataFrame()
        x = []
        y = []
        mapper_point_data_tag = ""
        for mapper_point_data_idx, mapper_point_data_value in mapper_point_data.items():
            mapper_point_data_tag = mapper_point_data_idx
            for in_range_idx in mapper_point_data[mapper_point_data_idx]:
                x.append(in_range_idx['x'])
                y.append(in_range_idx['y'])
                ideal_column_y = in_range_idx['ideal_column']
                dict = {'x': x, 'y': y}

        df_in_range_point = pd.DataFrame(dict)
        x = df_in_range_point['x'].to_numpy()
        y = df_in_range_point['y'].to_numpy()
        ideal_col_y = 0
        max_deviation = 0
        max_deviation_tag = ''
        for map_idx in max_deviation_train_ideal_dict:
            for ideal_idx in max_deviation_train_ideal_dict[map_idx]:
                ideal_col_y = ideal_idx
                max_deviation = max_deviation_train_ideal_dict[map_idx][ideal_idx]
                max_deviation_tag = f'{map_idx}_{ideal_idx}'

                x_data_pt = self.ideal_data_df['x'].to_numpy()
                y_data_pt = self.ideal_data_df[ideal_col_y].to_numpy()

                df = pd.DataFrame()  # Initialize df outside the if condition
                if (mapper_point_data_tag == max_deviation_tag):
                    df['x'] = x_data_pt
                    df['y'] = y_data_pt
                    ideal_slope, ideal_intercept, r_value, p_value, std_err = stats.linregress(df['x'], df['y'])
                    df['y_bestfit'] = (ideal_slope * df['x']) + ideal_intercept
                    df['y_upperband'] = df['y_bestfit'] + max_deviation
                    df['y_lowerband'] = df['y_bestfit'] - max_deviation
                    source = ColumnDataSource(df.reset_index())
                    p = figure()
                    p.scatter(x='x', y='y_bestfit', line_color=None, fill_alpha=0.5, size=5, source=source)
                    band = Band(base='x', lower='y_lowerband', upper='y_upperband', source=source, level='underlay', fill_alpha=1.0, line_width=1, line_color='black')

                    p.add_layout(band)
                    p.title.text = f"x vs {ideal_col_y}, max deviation :{max_deviation}"
                    p.xgrid[0].grid_line_color = None
                    p.ygrid[0].grid_line_alpha = 0.5
                    p.xaxis.axis_label = 'X'
                    p.yaxis.axis_label = 'Y'
                    size = 10
                    color = magma(256)
                    file_header = f"mapped_testdata_train_function_{map_idx}_ideal_function_{ideal_col_y}"
                    # generate legend for
                    p.scatter(x, y, size=size, color="red", legend_label="Mapped testdata within max deviation")
                    chart_filename = mapping_chart_save_path + file_header + ".html"
                    output_file(filename=chart_filename, title=file_header)
                    try:
                        show(row(p))
                        pass
                    except Exception as _error:
                        print(f"Error: GRAPH GENERATED UNSUCCESSFUL: {_error}")

    def generate_map_test_data_chart(self, map_data_set_dict, max_deviation_train_ideal_dict, mapping_chart_save_path):
        """
        Generate a map test data chart.

        Args:
            map_data_set_dict (dict): Map data set dictionary.
            max_deviation_train_ideal_dict (dict): Maximum deviation train ideal dictionary.
            mapping_chart_save_path (str): Path to save the chart.
        """
        max_deviation_tag_list = []
        for train_col_y in max_deviation_train_ideal_dict:
            for ideal_col_y in max_deviation_train_ideal_dict[train_col_y]:
                max_deviation_tag_list.append(f"{train_col_y}_{ideal_col_y}")
        for max_deviation_idx in max_deviation_tag_list:
            if max_deviation_idx not in map_data_set_dict:  # Add error handling
                print(f"Warning: Key {max_deviation_idx} not found in map_data_set_dict.")
                continue
            in_range_data_points_list = []
            in_range_data_points_dict = {}
            for in_range_max_max_deviation in map_data_set_dict[max_deviation_idx]:
                in_range_data_points = map_data_set_dict[max_deviation_idx][in_range_max_max_deviation]
                in_range_data_points_list.append(in_range_data_points)
                in_range_data_points_dict[max_deviation_idx] = in_range_data_points_list
            self.generate_mapper_graph(in_range_data_points_dict, max_deviation_train_ideal_dict, mapping_chart_save_path)


    def generate_max_deviation_graph(self, max_deviation_df, max_deviation, train_column_y, ideal_column_y, save_chart):
        """
        Generate a maximum deviation graph.

        Args:
            max_deviation_df (DataFrame): DataFrame containing maximum deviation data.
            max_deviation (float): Maximum deviation value.
            train_column_y (str): Train column y value.
            ideal_column_y (str): Ideal column y value.
            save_chart (str): Path to save the chart.
        """
        x_axis_list = []
        x_axis = max_deviation_df['x'].tolist()
        x_axis = list(map(str, x_axis))
        deviation = max_deviation_df['deviation'].tolist()
        for idx in range(0, len(max_deviation_df['deviation'])):
            x_axis_list.append(f"{deviation[idx]} : {x_axis[idx]}")

        color = ['blue'] * 400
        index_max_deviation = deviation.index(max(deviation))
        color[index_max_deviation] = 'red'
        TOOLS = "pan,box_zoom,reset,save,zoom_out,zoom_in"
        p = figure(y_range=FactorRange(factors=x_axis_list), max_width=5000, height=5000,title=f"Max Deviation : train dataset: {train_column_y} vs Ideal dataset{ideal_column_y}", toolbar_location=None, tools=TOOLS)
        p.hbar(y=x_axis_list, right=deviation, height=0.2, color=color)
        file_header = f"max_deviation_train_data_{train_column_y}_vs_ideal function_{ideal_column_y}"
        chart_filename = save_chart + file_header + ".html"
        output_file(filename=chart_filename, title=file_header)
        try:
            show(row(p))
            pass
        except Exception as _error:
            print(f"Error: GRAPH GENERATED UNSUCCESSFUL: {_error}")

    