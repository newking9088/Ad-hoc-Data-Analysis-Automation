# Ad-hoc-Data-Analysis-Automation
Use Python and SQL to automate data analysis by pulling data from AWS Redshift through a pipeline.

We use boto3 to pull data from AWS Redshift using Python and SQL. The pipeline leverages classes and Python tool functions to automate repeated ad-hoc tasks, including data processing, statistical analysis, visualization and report generation.

## Prerequisites
- Python 3.x
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Amazon Web Service (AWS) Command Line Service (CLI) (Edit your credentials in jupyter notebooks/db_credentials.json)
- Redshift (Data should be stored in Redshift if you do not have your data in .csv/.xlsx etc. file formats in AWS S3, local machine or elsewhere)
- tqdm
- boto3
- bz2
- json
- pingouin

If you have a running Python 3.x environment, you can simply use `pip install -r requirements.txt` to install all the required dependencies.
  
## Files and Directories

### python files/parse_compact.py

`parse_bz2` is a Python function designed to read and process compressed JSON data coming from Rockpi raw log messages data stored in `.bz2` file. It uses the `bz2`, `json`, and `pandas` libraries to transform the data into a structured DataFrame, making it easier to analyze and manipulate.

**Installation**
To use this function, ensure you have the following Python libraries installed:
- `bz2`
- `json`
- `pandas`

You can install `pandas` using pip:
```bash
pip install pandas
```

### python files/CycloidConnector.py
`VWCycloid` is a Python class designed to interact with an AWS Redshift database to retrieve and process cycloid data. It uses the `boto3` library to connect to Redshift and `pandas` to handle data manipulation.

**Installation**
To use this class, ensure you have the following Python libraries installed:
- `boto3`
- `pandas`
- `tqdm`

You can install these libraries using pip:
```bash
pip install boto3 pandas tqdm
```

### python files/adhoc_analysis.py
` adhoc` is a powerful Python class crafted to streamline and automate daily ad-hoc analysis requests from both internal and external clients. It offers a suite of methods designed to efficiently handle and simplify these tasks.

#### outlier_handler
**Purpose:** Handles outliers in a pandas Series by replacing them with the rolling median.

**Parameters:**
- `series (pd.Series)`: The input Series.
- `window (int)`: The window size for calculating the rolling statistics. Default is 5.
- `sigma_multiplier (float)`: Defines the boundary for outliers. Default is 0.1.

**Returns:** A `pd.Series` with outliers handled.

#### get_discharge_energy
**Purpose:** Retrieves discharge energy data, including median and lower bound energy, from a cycloid DataFrame.

**Parameters:**
- `df (pd.DataFrame)`: Input cycloid DataFrame.
- `window (int)`: The window size for calculating the rolling statistics. Default is 7.
- `sigma_multiplier (float)`: Defines the boundary for outliers. Default is 0.1.
- `disclaimer (bool)`: If True, prints a disclaimer about energy jumps. Default is True.
- `raw (bool)`: If True, returns raw discharge energy data without further processing. Default is False.

**Returns:** A `pd.DataFrame` with discharge energy data.

#### ttest
**Purpose:** Performs a t-test for A/B testing, comparing two samples (control and experiment) to return the p-value, practical impact factor (Cohen's d), and the power of the test.

**Parameters:**
- `sample1 (pd.Series, np.array, tuple, list)`: The first array-like or input sample.
- `sample2 (pd.Series, np.array, tuple, list)`: The second array-like or float input sample.
- `alternative (str, optional)`: Defines the alternative hypothesis or tail of the test. Must be one of “two-sided” (default), “greater”, or “less”. “greater” tests against the alternative hypothesis that the mean of sample1 is greater than the mean of sample2.
- `interprete (bool, optional)`: If True, prints an interpretation of the test results. Default is True.

**Returns:** A `pd.DataFrame` with the following statistical inference parameters:
- `p-val`: The p-value of the test.
- `cohen-d`: The practical impact factor.
- `power`: The power of the test.
- `BF10`: The Bayesian Factor supporting the alternative hypothesis.
- `CI95%`: The 95% confidence interval.

**Prints:** If `interprete` is True, prints an interpretation of the test results, including:
- Whether the difference in means is statistically significant.
- The practical impact of the experiment.
- The statistical power of the test.
- Bayesian support for the alternative hypothesis.

#### get_statistics_before_after_rpt
**Purpose:** Calculates statistics for cycles before and after a specified RPT (Reference Performance Test) cycle, providing insights into the performance changes.

**Parameters:**
- `df (pd.DataFrame)`: The input DataFrame containing cycloid data.
- `rpt_cycle (int)`: The cycle where the RPT is conducted. Default is 201.
- `neighbors_cycle (int)`: Number of cycles to consider before and after the RPT cycle. Default is 3.
- `window (int)`: Rolling window size for outlier handling. Default is 5.
- `sigma_multiplier (float)`: Sigma multiplier for outlier detection. Default is 0.1.
- `avoid_close_cycle (int)`: Number of cycles to avoid immediately before and after the RPT cycle. Default is 0.

**Returns:** A `pd.DataFrame` with statistics for each cycle around the specified RPT cycle, including:
- `min`: Minimum normalized discharge energy.
- `avg`: Average normalized discharge energy.
- `max`: Maximum normalized discharge energy.
- `std`: Standard deviation of normalized discharge energy.
- `cv_pc`: Coefficient of variation (percentage).
- `sample_size`: Number of samples in each cycle.

**Prints:** If applicable, prints the results of a t-test to determine if there is a significant increase in mean discharge energy after the RPT cycle, including:
- Sample sizes before and after the RPT.
- Statistical significance of the difference in means.
- Practical impact of the RPT on discharge energy.
- Bayesian support for the alternative hypothesis.
- Statistical power of the test.

#### get_energy_cycle_gain
**Purpose:** Calculates energy gains and statistics for each instance around a specified RPT (Reference Performance Test) cycle.

**Parameters:**
- `df (pd.DataFrame)`: The input DataFrame containing 'doe_number', 'instance_number', 'cycle', and 'adjusted_raw_discharge_energy' columns.
- `rpt_cycle (int)`: The reference cycle number where the RPT test is conducted. Default is 201.
- `neighbors_cycle (int)`: The number of cycles to consider before and after the RPT cycle for statistical calculations. Default is 5.
- `avoid_close_cycle (int)`: The number of cycles to avoid around the RPT cycle when calculating statistics. Default is 0.
- `window (int)`: The window size for the rolling function used in outlier detection. Default is 5.
- `sigma_multiplier (float)`: The multiplier for the standard deviation to determine outliers. Default is 0.1.
- `show_plot (bool)`: If True, displays a plot of the energy gains. Default is True.

**Returns:** A `pd.DataFrame` containing the computed statistics and energy gains for each instance, including:
- `avg_energy_before`: Average normalized discharge energy before the RPT cycle.
- `median_energy_before`: Median normalized discharge energy before the RPT cycle.
- `std_energy_before`: Standard deviation of normalized discharge energy before the RPT cycle.
- `avg_energy_after`: Average normalized discharge energy after the RPT cycle.
- `median_energy_after`: Median normalized discharge energy after the RPT cycle.
- `std_energy_after`: Standard deviation of normalized discharge energy after the RPT cycle.
- `gain_in_energy`: Gain in median normalized discharge energy after the RPT cycle.

**Prints:** If `show_plot` is True, displays a plot of the energy gains.

#### get_dcir
**Purpose:** Retrieves DCIR (Direct Current Internal Resistance) data from a cycloid DataFrame.

**Parameters:**
- `df (pd.DataFrame)`: Input cycloid DataFrame containing a 'dcir' column.

**Returns:** A `pd.DataFrame` with DCIR data, including:
- `doe_number`: DOE number.
- `instance_number`: Instance number.
- `cycle`: Cycle number.
- `dcir`: DCIR values, converted to milliohms and rounded to one decimal place.

#### plot_dcir
**Purpose:** Plots DCIR (Direct Current Internal Resistance) data over cycle numbers, with options for coloring and labeling.

**Parameters:**
- `df (pd.DataFrame)`: Input DataFrame from the `get_dcir()` method.
- `hue (str, optional)`: The column name to use for coloring the lines. Default is 'instance_number'.
- `label_title (str, optional)`: Title for the legend. Default is 'Instance Number'.
- `ax (matplotlib.axes.Axes, optional)`: Axes object to plot on. If not provided, a new figure will be created.

**Returns:** None. This method generates a plot of DCIR values over cycle numbers.

**Plot Details:**
- Uses a color palette to differentiate lines based on the `hue` parameter.
- Sets the y-axis label to "DCIR [mΩ]" and the x-axis label to "Cycle Number".
- Adds a legend with the specified `label_title`.

#### plot_mean_voltage
**Purpose:** Plots the mean voltage data for cycloid packs or cells over cycle numbers, with options for coloring, labeling, and outlier handling.

**Parameters:**
- `df (pd.DataFrame)`: Input DataFrame containing cycloid data.
- `hue (str, optional)`: The column name to use for coloring the lines. Default is 'instance_number'.
- `charging_state_name (str, optional)`: Substate for which to plot mean voltage. Default is 'CHARGE'.
- `label_title (str, optional)`: Title for the legend. Default is 'Instance Number'.
- `raw_data (bool, optional)`: If True, plots raw mean voltage data without cleaning outliers. Default is True.
- `window (int, optional)`: The window size for the rolling function used in outlier detection. Default is 5.
- `sigma_multiplier (float, optional)`: The multiplier for the standard deviation to determine outliers. Default is 0.2.
- `ax (matplotlib.axes.Axes, optional)`: Axes object to plot on. If not provided, a new figure will be created.

**Returns:** None. This method generates a plot of mean voltage values over cycle numbers.

**Plot Details:**
- Uses a color palette to differentiate lines based on the `hue` parameter.
- Filters data based on the specified `charging_state_name`.
- Optionally cleans outliers from the mean voltage data if `raw_data` is set to False.
- Sets the y-axis label to "Average Charge Voltage [V]" and the x-axis label to "Cycle Number".
- Adds a legend with the specified `label_title`.

#### plot_rover_metrics
**Purpose:** Plots up to 6 metrics in a subplot based on user input, allowing for detailed visualization of various metrics over time or cycles.

**Parameters:**
- `df (pd.DataFrame)`: The DataFrame containing the data to plot.
- `x (str, optional)`: The column name to use for the x-axis. Default is 'elapsed_minutes'.
- `charging_state_name (str, optional)`: The charging state to plot. Default is 'CHARGE'.
- `hue (str, optional)`: The column name to use for coloring the lines. Default is None.
- `suptitle (str, optional)`: Title for the entire plot. Default is None.
- `label_title (str, optional)`: Title for the legend labels. Default is None.
- `legend (bool, optional)`: Whether to show the legend for each subplot. Default is True.

**Returns:** None. This method generates a plot of up to 6 selected metrics in subplots.

**Plot Details:**
- Filters data based on the specified `charging_state_name`.
- Uses a color palette to differentiate lines based on the `hue` parameter.
- Ensures the current values are always positive.
- Prompts the user to select up to 6 metrics to plot from a list of available metrics.
- Creates subplots for the selected metrics, with a maximum of 6 metrics per plot.
- Sets appropriate labels and titles for the plots and legend.

#### read_data
**Purpose:** Reads data from a specified file path, supporting both CSV and Excel file formats, and renames certain columns for consistency.

**Parameters:**
- `data_file_path (str)`: The path to the data file, which should be either a .csv or .xlsx file.

**Returns:** A `pd.DataFrame` containing the data from the file, with columns renamed to 'doe', 'instance', 'cycle_num', and 'charging_state_name'.

**Prints:** 
- Unique DOE values in the data.
- Unique instance values in the data.
- Either the list of cycles if there are 6 or fewer, or the range of cycles if there are more than 6.

#### filter_by_doe
**Purpose:** Filters the DataFrame by the provided list of DOE values.

**Parameters:**
- `df (pd.DataFrame)`: The input DataFrame.
- `doe_list (list)`: The list of DOE values to filter by.

**Returns:** A `pd.DataFrame` filtered to include only the rows with DOE values in `doe_list`.

#### filter_by_instance
**Purpose:** Filters the DataFrame by the provided list of instance values.

**Parameters:**
- `df (pd.DataFrame)`: The input DataFrame.
- `instance_list (list)`: The list of instance values to filter by.

**Returns:** A `pd.DataFrame` filtered to include only the rows with instance values in `instance_list`.

#### filter_by_cycle
**Purpose:** Filters the DataFrame by the provided list of cycle numbers.

**Parameters:**
- `df (pd.DataFrame)`: The input DataFrame.
- `cycle_list (list)`: The list of cycle numbers to filter by.

**Returns:** A `pd.DataFrame` filtered to include only the rows with cycle numbers in `cycle_list`.

#### filter_by_doe_instance
**Purpose:** Filters the DataFrame by the provided list of (DOE, instance) tuples.

**Parameters:**
- `df (pd.DataFrame)`: The input DataFrame.
- `doe_instance_tuple (list of tuples)`: List of tuples containing (doe, instance) values to filter by.

**Returns:** A `pd.DataFrame` filtered to include only the rows with matching (doe, instance) values from `doe_instance_tuple`.

#### get_cells_voltage_std
**Purpose:** Calculates the standard deviation of voltage for 5 cells in a pack for each cycle and cycling substate, and returns a dictionary with the cycling substate as the key and the corresponding DataFrame as the value.

**Parameters:**
- `df (pd.DataFrame)`: Input cycloid DataFrame with columns for cells' min and max voltage.
- `window (int, optional)`: The window size for the rolling function used in outlier detection. Default is 5.
- `sigma_multiplier (Union[int, float], optional)`: The multiplier for the standard deviation to determine outliers. Default is 0.1.

**Returns:** A dictionary where each key is a cycling substate and each value is a DataFrame with the standard deviation of cells' voltage added for each cycle and substate.

#### plot_cells_voltage_std
**Purpose:** Plots the mean and 95% confidence interval of the standard deviation of 5 cells' voltage versus cycle number for each cycling substate.

**Parameters:**
- `df (pd.DataFrame)`: Input DataFrame, typically the output from the `get_cells_voltage_std()` method.

**Returns:** None. This method generates and displays a plot.

**Plot Details:**
- Uses a color palette to differentiate lines.
- Sets common y-axis limits for each cycling substate.
- Adds appropriate labels and titles to the plots.
- Displays the plot with mean and 95% confidence interval of the standard deviation of cells' voltage.

#### plot_cells_voltage
**Purpose:** Plots voltage data for different cycling states, providing a visual representation of cell voltages over cycles for a specific instance.

**Parameters:**
- `df (pd.DataFrame)`: Input cycloid DataFrame containing voltage data.
- `instance_number (Union[int, str], optional)`: Instance number to plot. If None, a random instance is selected. Default is None.
- `raw_data (bool, optional)`: If True, plots raw data. If False, plots cleaned data with outliers handled. Default is True.
- `window (int, optional)`: Window size for rolling median and standard deviation used in outlier detection. Default is 5.
- `sigma_multiplier (Union[float, int], optional)`: Sigma multiplier for outlier detection. Default is 0.1.

**Returns:** None. This method generates and displays a 2x2 plot with cell voltages for each cycling substate.

**Plot Details:**
- Filters the DataFrame by the specified instance number.
- Defines voltage column mappings for different cycling states.
- Uses a color palette to differentiate lines for each cell.
- Optionally cleans outliers from the voltage data if `raw_data` is set to False.
- Creates subplots for each cycling substate (CHARGE, CHARGE REST, DISCHARGE, DISCHARGE REST).
- Sets appropriate labels and titles for the plots and legend.
- Displays the plot with cell voltages over cycle numbers.

#### get_polarization_voltage
**Purpose:** Calculates the polarization voltage within each cycling substate by computing the difference between the maximum and minimum voltage for each cell and the pack.

**Parameters:**
- `df (pd.DataFrame)`: Input cycloid DataFrame.

**Returns:** A `pd.DataFrame` with the following columns:
- `doe_number`: DOE number.
- `instance_number`: Instance number.
- `cycle`: Cycle number.
- `cycling_substate`: Cycling substate.
- `cell{i}_delta_voltage`: Difference between the maximum and minimum voltage for each cell (i = 1 to 5).
- `pack_delta_voltage`: Difference between the maximum and minimum pack voltage.

#### plot_polarization_voltage
**Purpose:** Plots the polarization voltage for both the pack and individual cells over cycle numbers, with options for outlier handling and legend display.

**Parameters:**
- `df (pd.DataFrame)`: Input DataFrame.
- `window (int, optional)`: The window size for the rolling function used in outlier detection. Default is 5.
- `sigma_multiplier (Union[int, float], optional)`: The multiplier for the standard deviation to determine outliers. Default is 0.1.
- `raw_data (bool, optional)`: If True, plots raw data. If False, plots cleaned data with outliers handled. Default is True.
- `legend (bool, optional)`: Whether to show the legend for each subplot. Default is True.

**Returns:** None. This method generates and displays plots for the polarization voltage of the pack and individual cells.

**Plot Details:**
- Filters data for 'CHARGE REST' and 'DISCHARGE REST' cycling substates.
- Uses a color palette to differentiate lines based on the `instance_number`.
- Optionally cleans outliers from the polarization voltage data if `raw_data` is set to False.
- Creates subplots for pack and individual cell polarization voltages.
- Sets appropriate labels and titles for the plots and legend.
- Adjusts y-axis limits to be consistent across subplots.
- Displays the plot with polarization voltages over cycle numbers.


#### plot_temperature
**Purpose:** Plots temperature data for different cycling states, providing a visual representation of temperature changes over cycles for a specific instance.

**Parameters:**
- `df (pd.DataFrame)`: Input DataFrame containing temperature data.
- `x (str, optional)`: The column name to use for the x-axis. Default is 'cycle'.
- `y (str, optional)`: The column name to use for the y-axis. Default is 'max_temp'.
- `hue (str, optional)`: The column name to use for coloring the lines. Default is 'instance_number'.
- `clean_first_n_cycles (int, optional)`: Number of initial cycles to clean for temperature spikes. Default is 5.
- `window (int, optional)`: Window size for the rolling function used in outlier detection. Default is 7.
- `sigma_multiplier (Union[int, float], optional)`: The multiplier for the standard deviation to determine outliers. Default is 0.5.
- `sharex (bool, optional)`: Whether to share the x-axis among subplots. Default is True.

**Returns:** None. This method generates and displays a 2x2 plot with temperature data for each cycling substate.

**Plot Details:**
- Filters the DataFrame by the specified cycling substates ('CHARGE', 'CHARGE REST', 'DISCHARGE', 'DISCHARGE REST').
- Cleans initial cycles for temperature spikes by replacing values with the median.
- Optionally cleans outliers from the temperature data if `raw_data` is set to False.
- Uses a color palette to differentiate lines based on the `hue` parameter.
- Creates subplots for each cycling substate.
- Sets appropriate labels and titles for the plots and legend.
- Displays the plot with temperature data over cycle numbers.

### jupyter notebooks/adhoc_worksheet.ipynb

This section demonstrates how to use the provided methods for analyzing and visualizing cycloid data in a Jupyter Notebook.

**User Input**
```python
# User input
doe_number = 'MTSOWPhase2_Pack25CValidation'  # Provide DOE as int or str
instance_number = None  # Provide a list of instances
cycles = None  # Provide a list of cycles
default_columns = True  # If True, pulls all the columns
override_query = None  # You can write your own SQL query if you like

# Fetch cycloid data from Redshift
dfc = redshift.get_cycloid_data(doe=doe_number, instances=instance_number, 
                                cycles=cycles, default_columns=default_columns, 
                                override_query=override_query)

# Use the provided methods to get DCIR and discharge energy data
dcir = adhoc.get_dcir(dfc)
discharge_energy = adhoc.get_discharge_energy(dfc)
```

Please refer to adhoc_worksheet.ipynb to learn how to implement the available methods for automating ad-hoc data analysis. The notebook provides detailed explanations, examples, and outputs to guide you through the process.

### jupyter notebooks/dump_data_into_psql.ipynb

#### Background
Sometimes we need to dump data into our Redshift database from an external source. After implementing new software or hardware changes, for example, the data might be messy. We can clean the data using Python, convert it into CSV or XLSX format, and then dump it into the Redshift database. This function automates that task.

 #### DBCredentials**
**Purpose:** A dataclass to store database credentials.

**Attributes:**
- `user (str)`: Username for the database.
- `password (str)`: Password for the database.
- `host (str)`: Host address of the database.
- `port (int)`: Port number to connect to the database.
- `dbname (str)`: Name of the database.

#### load_credentials
**Purpose:** Loads database credentials from a JSON file.

**Parameters:**
- `json_file (str)`: Path to the JSON file containing the credentials.

**Returns:** An instance of the `DBCredentials` dataclass.

#### check_database_exists
**Purpose:** Checks if a database already exists.

**Parameters:**
- `creds (DBCredentials)`: The database credentials.

**Returns:** `bool`: True if the database exists, False otherwise.

#### create_db_and_dump_data
**Purpose:** Creates a database (if it doesn't already exist) and dumps data into a table.

**Parameters:**
- `creds (DBCredentials)`: The database credentials.
- `table_name (str)`: The name of the table to create or append data to.
- `data_file (str)`: Path to the data file (Excel or CSV) to be dumped into the table.
- `if_exists (str)`: What to do if the table already exists. Options are 'replace' or 'append'.

**Returns:** None. This function automates the process of creating a database and dumping cleaned data into it.

## jupyter notebooks/error_in_current_0.5_6A.ipynb

### CapacityTimeUncertainty
**Purpose:** This class is designed to handle and analyze cycloid data, with a specific focus on calculating and managing capacity and time uncertainties. These calculations are crucial for defining product specifications before market rollout, ensuring reliability and performance.

**Attributes:**
- `data (Union[pd.DataFrame, str])`: The input data, which can be a DataFrame or a file path to a CSV/XLSX file.
- `json_params (str)`: Path to the JSON file containing parameters. Default is 'params.json'.
- `df (pd.DataFrame)`: The DataFrame to store the processed data.
- `params (Dict[str, Union[float, Dict[str, float]])`: Dictionary to store parameters loaded from the JSON file.

#### __post_init__
**Purpose:** Initializes the class by loading parameters and data.

#### load_params
**Purpose:** Loads parameters from a JSON file.

**Parameters:**
- `json_params (str)`: Path to the JSON file containing the parameters.

#### load_data
**Purpose:** Loads data from a CSV/XLSX file or directly from a DataFrame, filters it for the 'CHARGE' state, and applies error limits.

**Parameters:**
- `data (Union[pd.DataFrame, str])`: The input data, which can be a DataFrame or a file path to a CSV/XLSX file.

#### apply_error_limits
**Purpose:** Applies error limits to the average current data based on parameters, and calculates the lower and upper bounds for the average current.

#### calculate_capacity_bounds
**Purpose:** Calculates the capacity bounds (lower, upper, and actual) for each cycle and instance.

#### get_clean_rover_data
**Purpose:** Cleans the data by removing cycles with capacity values outside the specified sigma bounds.

#### get_cycloid_capacity_err
**Purpose:** Aggregates the capacity data, calculates the capacity bounds and uncertainties, and removes outliers.

**Returns:** A DataFrame with the aggregated capacity data, including lower and upper bounds and uncertainties.

### jupyter notebooks/error_measurement_estimation.ipynb
It contains some detailed data cleaning and preprocessing before doing CapacityTimeUncertainty calculation.

### jupyter notebooks/pack_voltage_capacity_per_current_step.ipynb
It details why we should use different error limits on different regime of applied current for charge time and capacity uncertainty estimation when cycling a battery.

### data
This dataset includes both raw and cleaned data for battery cycling, intended for demonstration purposes. Please note that the data has been dummified and synthesized, and does not represent real-world data.










