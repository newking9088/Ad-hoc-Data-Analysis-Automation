# def import_or_install(package):
#     import importlib
#     try:
#         importlib.import_module(package)
#     except ImportError:
#         print(f"{package} is not installed. Installing ....")
#         import subprocess
#         subprocess.check_call(['pip', 'install', package])
# import_or_install('boto3')
# import_or_install('pandas')
# import_or_install('time')
# import_or_install('tqdm')
import boto3
import pandas as pd
from time import sleep, time, localtime
from tqdm import tqdm 

class VWCycloid: 

    def get_cycloid_data(self, doe : int or str, instances : list = None, cycles: list = None, 
        default_columns : list = False, override_query = None) -> pd.DataFrame:

        # Get the list of columns to pull 
        all_columns = ['doe_number', 'instance_number', 'cycle', 'cycling_substate',
       'start_time', 'end_time', 'capacity', 'soc_5_min', '"time"', 'energy',
       'energy_rest', 'capacity_charge_cc',
       'capacity_charge_cv', 'time_charge_cc', 'time_charge_cv', 
       'energy_charge_cc', 'energy_charge_cv', 'avg_temp',
       'max_temp', 'min_temp', 'mean_mean_voltage', 'min_mean_voltage',
       'max_mean_voltage', 'mean_min_voltage', 'min_min_voltage',
       'max_min_voltage', 'mean_max_voltage', 'min_max_voltage',
       'max_max_voltage', 'mean_mean_current', 'min_mean_current',
       'max_mean_current', 'mean_min_current', 'min_min_current',
       'max_min_current', 'mean_max_current', 'min_max_current',
       'max_max_current', 'avg_sum_cell_voltage_v', 'avg_cell1_voltage_v',
       'min_cell1_voltage_v', 'max_cell1_voltage_v', 'avg_cell2_voltage_v',
       'max_cell2_voltage_v', 'min_cell2_voltage_v', 'avg_cell3_voltage_v',
       'min_cell3_voltage_v', 'max_cell3_voltage_v', 'avg_cell4_voltage_v',
       'min_cell4_voltage_v', 'max_cell4_voltage_v', 'avg_cell5_voltage_v',
       'min_cell5_voltage_v', 'max_cell5_voltage_v', 'avg_cell6_voltage_v',
       'min_cell6_voltage_v', 'max_cell6_voltage_v', 'avg_rest_current_a',
       'min_rest_current_a', 'max_rest_current_a', 'avg_rest_voltage_v',
       'min_rest_voltage_v', 'max_rest_voltage_v', 'dcir']

        important_columns = ['doe_number', 'instance_number', 'cycle', 'cycling_substate', 'energy', 
        'energy_rest', '"time"', 'mean_mean_voltage', 'dcir', 'min_temp']

        columns = all_columns if default_columns else important_columns
        
        # AND (cycling_substate in ('CHARGE', 'CHARGE REST', 'DISCHARGE', 'DISCHARGE REST', 'DCIR'))

        # sql query from "vw_cycloid" view in "cycles" database 
        query = f"SELECT {','.join(columns)} FROM cycles.public.vw_cycloid  WHERE ((doe_number = '{doe}')\
        AND (cycle != 0))"

        if instances is not None:
            instances = [int(instance) if str(instance).isdigit() else str(instance) for instance in instances]
            instance_filter = ' AND instance_number in (' + ','.join([f"{i}" for i in instances]) + ')'
            query += instance_filter 
        if cycles is not None:
            cycles_filter = ' AND cycle in (' + ','.join([f"{i}" for i in cycles]) + ')'
            query += cycles_filter

        if not override_query is None: 
            query = override_query

        # Establish connection to redshift using boto3. The AWS SDK for Python 
        # (boto3) provides a Python API for AWS infrastructure services. 
        
        # create boto3 client for redshift data API which allows to interact with
        # the redshift cluster using Data API
        client = boto3.client("redshift-data")
        
        # Run a SQL query on the your Redshift cluster ('ClusterIdentifier') with
        # the specified database ('Database') and user ('DbUser') and store the result
        # in the variable 'response'
        # Execute the parameterized query using the provided parameters
        response = client.execute_statement(
            ClusterIdentifier = 'warehouse-warehouse-warehouse',
            DbUser = 'admin',
            Database = 'cycles',
            Sql = query   
              )

        # The response object is dictionary-like structure which contains information
        # about the execution of the SQL statement. It contain 'Id', 'Status', 'Error'
        # etc., and if the execution was successful return 'Result'.
        if response['Id']:
            last_status = None 
            time_out_minutes = 3 
            query_start_time = time() 

            # Wait for query to finish 
            while True:

                describe = client.describe_statement(Id = response['Id']) 
                status = describe['Status'] 

                if status != last_status:
                    print(f"Status: {status}")
                    last_status = status 

                if status in ['ABORTED','FAILED']:
                    print(f"EXITING QUERY DUE TO STATUS: {status}.")
                    break 
                
                if status == 'FINISHED':
                    print("QUERY FINISHED.")
                    break

                if (time() - query_start_time) / 60 > time_out_minutes:
                    print("EXITING QUERY DUE TO TIMEOUT.")
                    break 

                sleep(2)
            

            describe = client.describe_statement(Id = response['Id'])  
            result_rows = describe['ResultRows'] 

            # Process Query Results 
            with tqdm(desc = F'Aggregating Results for DoE {doe}', total = result_rows, unit = 'lines') as progress:
                
                # Check for Error 
                assert not 'Error' in client.describe_statement(Id = response['Id']), client.describe_statement(Id = response['Id'])['Error']
            
                # Query Result 
                result = client.get_statement_result(Id = response['Id'])
                result_data = self.process_result(result) 

                # Initialize table data 
                table_data = result_data

                progress.update(len(result_data))

                # Process Succeeding Pages 
                while 'NextToken' in result:

                    # Query Result (next page)
                    result = client.get_statement_result(Id = response['Id'], NextToken = result['NextToken'])
                    result_data = self.process_result(result)
                    
                    # Append to table data 
                    table_data += result_data

                    progress.update(len(result_data))

                # Aggregate Results into DataFrame 
                df = pd.DataFrame(table_data)

                # Convert instance_number column to int if it contains digits else str
                if 'instance_number' in df.columns:
                    df['instance_number'] = df['instance_number'].apply(lambda x: int(x) if str(x).isdigit() else str(x))

                if override_query is None:
                    df.sort_values(['doe_number','instance_number', 'cycle'], inplace=True)
                
                print("\nFrame created successfuly.")
                return df 
        else:
            print("No reponse Id.")
            return pd.DataFrame()

    def process_result(self, result): 
        """
        Process client.get_statement_result
        """
        
        def is_numeric_type(t):
            t = t.lower() 
            if 'float' in t or 'int' in t or 'numeric' in t:
                return True 
            return False

        def process_line(record_line, column_names, column_types):
            """
            Process a line from result['Records'] 

            Parameters: 
            -----------
            record_line : list of {'<bullshit>':value} 
            column_names : list of column names 
            column_types : list of column types 

            Output:
            -------

            List of cycloid data in the form: {'column_name':value, ...}
            """
            
            assert len(record_line) == len(column_names), "Line length does not match column length"
            assert len(column_names) == len(column_types), "Column length does not match type length"


            line_data = {} 

            for datum, column_name, column_type in zip(record_line, column_names, column_types): 
                
                # TODO ? This script assumes the datum (element of Line in Records) has only one value. When will it have not exactly 1? 
                assert len(datum) == 1, "This script has not implemented handling of record values not having exactly 1 value. Data for {column_name} as datum:\n\n{datum}\n\n"

                datum_type = list(datum.keys())[0] 
                datum_value = list(datum.values())[0] 

                # Handle Null Value
                if datum_type == 'isNull':
                    datum_value = None 
                
                # Handle Numeric String Types  
                elif is_numeric_type(column_type) and isinstance(datum_value, str): 
                    if datum_value.isnumeric():
                        datum_value = int(datum_value) 
                    else:
                        datum_value = float(datum_value) 
                
                # Append data to line data 
                line_data[column_name] = datum_value 
            
            return line_data 

        # Result Metadata 
        column_names = [column_data['name'] for column_data in result['ColumnMetadata']]      # Column Names 
        column_types = [column_data['typeName'] for column_data in result['ColumnMetadata']]  # Column Types 

        record_data = []
        for line in result['Records']:
            record = process_line(line, column_names, column_types)
            record_data.append(record) 
    
        return record_data
    