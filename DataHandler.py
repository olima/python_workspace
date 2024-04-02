from sqlalchemy import create_engine, inspect, text, delete
from sqlalchemy import Column, Float, String, Integer, MetaData, Table
from sqlalchemy.orm import sessionmaker
import pandas as pd

class DataAccess:
    """
    DataAccess Class

    This class provides functionality for accessing and managing data in a SQLite database.

    Attributes:
        table_name_dict (dict): Dictionary mapping internal table names to corresponding database table names.
        df_dict (dict): Dictionary containing DataFrames for different types of data.
        db_name (str): Name of the SQLite database.
        engine (sqlalchemy.engine.base.Engine): SQLAlchemy engine object for database connection.
        Session (sqlalchemy.orm.session.sessionmaker): Session class for interacting with the database.
    """
    def __init__(self, db_name):
        """
        Initialize the DataAccess class with the specified database name.

        Args:
            db_name (str): Name of the SQLite database.
        """
        self.table_name_dict = {"_test_data": "test_data_tb", "_train_data": "train_data_tb", "_ideal_data": "ideal_data_tb"}
        self.df_dict = {"_test_data": '', "_train_data": '', "_ideal_data": ''}
        self.db_name = db_name
        try:
            self.engine = create_engine(f'sqlite:///{db_name}', echo=True)
            self.Session = sessionmaker(bind=self.engine)
        except Exception as _error:
            print(f"Error: DATABASE ENGINE COULD NOT CREATED: {_error}")

    def load_csv_to_db(self, train_data_csv, ideal_data_csv, test_data_csv, db_path):
        """
        Load CSV files into the database.

        Args:
            train_data_csv (str): Path to the CSV file containing training data.
            ideal_data_csv (str): Path to the CSV file containing ideal data.
            test_data_csv (str): Path to the CSV file containing test data.
            db_path (str): Path to the SQLite database.
        """
        self.ideal_data_csv = ideal_data_csv
        self.train_data_csv = train_data_csv
        self.test_data_csv = test_data_csv
        self.db_path = db_path
        session = self.Session()
        self.update_rawdata_to_db(self.ideal_data_csv, self.train_data_csv, self.test_data_csv)
        self._create_table_from_df()
        session.commit()
        session.close()

    def update_rawdata_to_db(self, ideal_data_csv, train_data_csv, test_data_csv):
        """
        Update raw data from CSV files into DataFrames.

        Args:
            ideal_data_csv (str): Path to the CSV file containing ideal data.
            train_data_csv (str): Path to the CSV file containing training data.
            test_data_csv (str): Path to the CSV file containing test data.
        """
        self.df_dict["_ideal_data"] = pd.read_csv(ideal_data_csv)
        self.df_dict["_train_data"] = pd.read_csv(train_data_csv)
        self.df_dict["_test_data"] = pd.read_csv(test_data_csv)
        print(self.df_dict["_ideal_data"])
        print(self.df_dict["_train_data"])
        print(self.df_dict["_test_data"])

    def _create_table_from_df(self):
        """Create tables from DataFrames."""
        inspect_db = inspect(self.engine)
        table_names_db = inspect_db.get_table_names()
        for table_name in table_names_db:
            if table_name in list(self.table_name_dict.values()):
                sql_cmd = text(f"DROP TABLE {table_name}")
                with self.engine.connect() as connection:  # Establish connection
                    connection.execute(sql_cmd)
        with self.engine.connect() as connection:
            self.df_dict["_ideal_data"].to_sql(self.table_name_dict["_ideal_data"], self.engine)
            self.df_dict["_train_data"].to_sql(self.table_name_dict["_train_data"], self.engine)
            self.df_dict["_test_data"].to_sql(self.table_name_dict["_test_data"], self.engine)

    def get_test_data_df(self):
        """
        Retrieve test data DataFrame from the database.

        Returns:
            DataFrame: DataFrame containing test data.
        """
        session = self.Session()
        test_data_tb = pd.read_sql_table('test_data_tb', self.engine)
        session.close()
        return test_data_tb

    def get_ideal_data_df(self):
        """
        Retrieve ideal data DataFrame from the database.

        Returns:
            DataFrame: DataFrame containing ideal data.
        """
        session = self.Session()
        test_ideal_tb = pd.read_sql_table('ideal_data_tb', self.engine)
        session.close()
        return test_ideal_tb

    def get_train_data_df(self):
        """
        Retrieve train data DataFrame from the database.

        Returns:
            DataFrame: DataFrame containing training data.
        """
        session = self.Session()
        test_train_tb = pd.read_sql_table('train_data_tb', self.engine)
        session.close()
        return test_train_tb

    def create_tb(self, db_table_name, data_df):
        """
        Create a table in the database with the given name and DataFrame.

        Args:
            db_table_name (str): Name of the table to be created.
            data_df (DataFrame): DataFrame containing the data for the table.
        """
        try:
            data_df.to_sql(db_table_name, self.engine)
        except Exception as _error:
            print(f"Error: UNABLE TO CREATE TABLE : {_error}")

    def create_mapping_test_data_table(self, mapping_test_data_dict):
        """
        Create a mapping table for test data.

        Args:
            mapping_test_data_dict (dict): Dictionary containing mapping data for test data.
        """
        session = self.Session()
        metadata = MetaData()
        mapping_test_data_table = Table('MappingTestData_tb', metadata,
                                        Column('id', Integer, primary_key=True),
                                        Column('max_deviation_idx', String),
                                        Column('ideal_column', String),
                                        Column('x', Float),
                                        Column('y', Float),
                                        Column('y_upper_band', Float),
                                        Column('y_lower_band', Float))
        metadata.create_all(self.engine)
        session.execute(delete(mapping_test_data_table))
        for key, value in mapping_test_data_dict.items():
            max_deviation_idx, ideal_column = key.split('_')
            for _, data_point in value.items():
                x = data_point['x']
                y = data_point['y']
                y_upper_band = data_point['y_upperband']
                y_lower_band = data_point['y_lowerband']
                session.execute(mapping_test_data_table.insert().values(
                    max_deviation_idx=max_deviation_idx,
                    ideal_column=ideal_column,
                    x=x,
                    y=y,
                    y_upper_band=y_upper_band,
                    y_lower_band=y_lower_band
                ))
        session.commit()
        session.close()

    def get_mapping_test_data_table(self):
        """
        Retrieve mapping test data table from the database.

        Returns:
            DataFrame: DataFrame containing mapping test data.
        """
        session = self.Session()
        query = text("SELECT * FROM MappingTestData_tb")
        mapping_data = session.execute(query).fetchall()
        mapping_data_df = pd.DataFrame(mapping_data,
                                    columns=['id', 'max_deviation_idx', 'ideal_column', 'x', 'y', 'y_upper_band', 'y_lower_band'])
        session.close()
        return mapping_data_df