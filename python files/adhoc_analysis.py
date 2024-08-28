import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from collections.abc import Iterable
import warnings
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator
from typing import Union, List, Tuple
import pingouin as pg
sns.set_style("whitegrid")
warnings.filterwarnings('ignore') 

# Update rcParams in mpl
rcParams = {
        'font.size': 20,                # all fonts-size in plot
        'font.weight': 'bold',          # bold all fonts
        'figure.titleweight': 'bold',   # bold supertitle
        'axes.linewidth' : 6,
        'xtick.major.width': 6,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'lines.linewidth': 6,
        'legend.fontsize': 'large',
        'xtick.labelsize': 'large',
        'ytick.labelsize': 'large',
        'xtick.labelsize': 20,   # sets x-tick font size
        'ytick.labelsize': 20,
        'axes.titlepad': 20,   # spacing between suptitle and figure
        'axes.facecolor': 'white'
    }

# Update rcparams
mpl.rcParams.update(rcParams) 

class Adhoc:
    def __init__(self):
        self.sns = sns
        self.plt = plt

    def outlier_handler(self, series: pd.Series, window: int = 5, sigma_multiplier : float = 0.1) -> pd.Series:
        """
        Handle outliers in a pandas Series.

        Parameters:
            series (pd.Series): The input Series.
            sigma_multiplier (float): Defines the boundary for outliers.

        Returns:
            pd.Series: Series with outliers handled.
        """
        roll = series.rolling(window = window, min_periods = 1)
        md = roll.median()
        sd = roll.std(ddof = 0)
        z_score = (series - md) / sd
        series[(z_score < - sigma_multiplier) | (z_score > sigma_multiplier)] = md
        return series
    
    def get_discharge_energy(self, df: pd.DataFrame, window: int = 7, sigma_multiplier : float = 0.1, 
                             disclaimer: bool = True, raw: bool = False) -> pd.DataFrame:
        """
        Get discharge energy with median and lower bound energy.

        Parameters:
            df (pd.DataFrame): Input cycloid DataFrame.

        Returns:
            pd.DataFrame: DataFrame with discharge energy data.
        """

        # sort the values by 'doe_number', 'instance_number', 'cycle'
        df.sort_values(by = ['doe_number', 'instance_number', 'cycle'], inplace = True)

        # Handle the first 5 cycles outlier separately
        def adjust_energy(group):
            pct_change = group.loc[group['cycle'] < 6, 'raw_discharge_energy'].pct_change().abs()
            size = len(pct_change)
            if (pct_change > 0.005).any():
                mean = group.loc[group['cycle'] < 6, 'raw_discharge_energy'].median()
                scale = mean * 0.00025
                group.loc[group['cycle'] < 6, 'raw_discharge_energy'] = sorted(np.random.normal(loc = mean, scale = scale, size = size))[::-1]
            return group

        substates = df['cycling_substate'].unique()
        if 'DCIR' not in substates:
            discharge_energy_df = df.loc[df['cycling_substate'] == 'DISCHARGE', ['doe_number', 'instance_number',
                           'cycle', 'cycling_substate', 'energy']].reset_index(drop = True)
            discharge_energy_df.rename(columns = {'energy': 'raw_discharge_energy'}, inplace = True)
        else:
            df = df.loc[df['cycling_substate'].isin(['DISCHARGE', 'DCIR']), ['doe_number', 'instance_number',
                           'cycle', 'cycling_substate', 'energy', 'energy_rest']]
            discharge_energy_df = df.groupby(['doe_number', 'instance_number', 'cycle']).apply(lambda group: 
            pd.Series({'raw_discharge_energy': group.loc[group['cycling_substate'] == 'DISCHARGE', 'energy'].sum(skipna = True)
             + group.loc[group['cycling_substate'] == 'DCIR', 'energy_rest'].sum(skipna = True)})).reset_index()
            # Create adjusted (Handle the first 5 cycles outlier separately) raw discharge energy as well
            discharge_energy_df['adjusted_raw_discharge_energy'] = discharge_energy_df.groupby(['doe_number', 'instance_number'])\
            .apply(adjust_energy)['raw_discharge_energy'].values

        if raw:
            return discharge_energy_df
        
        if disclaimer:
            print("Disclaimer: Since the normalized energy data are cleaned, if there is an energy jump")
            print("after RPT test or due to days/weeks pause, the jump in energy may be delayed by window size")
            print("that was specified in outlier handler function.")
            
        # Apply outlier handler
        discharge_energy_df['discharge_energy'] = discharge_energy_df.groupby(['doe_number', 'instance_number'])['adjusted_raw_discharge_energy']\
        .transform(lambda x: self.outlier_handler(x, window = window, sigma_multiplier = sigma_multiplier))
        
        # Calculate median energy
        discharge_energy_df['mean_energy'] = discharge_energy_df.groupby(['doe_number', 'cycle'])['discharge_energy'].transform('median')
        
        # Calculate standard deviation of energy
        discharge_energy_df['std_energy'] = discharge_energy_df.groupby(['doe_number', 'cycle'])['discharge_energy'].transform('std')
        
        # Calculate lower energy bound
        discharge_energy_df['lower_energy_bound'] = discharge_energy_df['mean_energy'] - 3 * discharge_energy_df['std_energy']

        # Get normalized discharge energy
        discharge_energy_df['norm_discharge_energy'] = discharge_energy_df['discharge_energy']/(discharge_energy_df.groupby
        (['doe_number', 'instance_number'])['discharge_energy'].transform(lambda x: x.iloc[1]))
        discharge_energy_df['norm_discharge_energy'] = discharge_energy_df.groupby(['doe_number', 'instance_number'])\
        ['norm_discharge_energy'].apply(lambda x: self.outlier_handler(x, window = window, sigma_multiplier = sigma_multiplier)).values

        discharge_energy_df['norm_mean_energy'] = discharge_energy_df['mean_energy']/(discharge_energy_df.groupby
        (['doe_number', 'instance_number'])['mean_energy'].transform(lambda x: x.iloc[1]))

        discharge_energy_df['norm_lower_energy'] = discharge_energy_df['lower_energy_bound']/(discharge_energy_df.groupby
        (['doe_number', 'instance_number'])['lower_energy_bound'].transform(lambda x: x.iloc[1]))

        return discharge_energy_df
    
    def ttest(self, sample1, sample2, alternative = 'two-sided', interprete = True):
        """
        Takes two samples, control and experiment, for t-test and returns p-value,
        practical impact factor cohen-d, power of the test.

        Parameters:
        sample1 (pd.Series, np.array, tuple, list): The first array-like or input sample.
        sample2 (pd.Series, np.array, tuple, list): The second array-like or float input sample.
        alternative (optional, str): Defines the alternative hypothesis, or tail of the test.
        Must be one of “two-sided” (default), “greater” or “less”. Both “greater” and “less” return 
        one-sided p-values. “greater” tests against the alternative hypothesis that the 
        mean of sample1 is greater than the mean of sample2.

        Returns:
        pd.DataFrame: DataFrame with statistical inference parameters.
        """
        if len(sample1) > 1 and len(sample2) > 1:
            unpaired_ttest = pg.ttest(sample1, sample2, alternative = \
            alternative, correction = True).reset_index(drop = True)

            if interprete:
                p_value = float(unpaired_ttest.at[0, 'p-val'])
                cohen_d = float(abs(unpaired_ttest.at[0, 'cohen-d']))
                ci = unpaired_ttest.at[0, 'CI95%']
                power_of_test = float(unpaired_ttest.at[0, 'power'])
                bayesian_support_alternative = float(unpaired_ttest.at[0, 'BF10'])

                significance = 'statistically significant' if p_value < 0.05 else 'not statistically significant'
                supports = 'supports the idea that mean before and after changes significantly'\
                if bayesian_support_alternative >= 1.0 else 'supports the idea that mean before and after\
                does not change significantly'
                effect_size = ('a very small impact' if cohen_d < 0.2 else
                                'a small impact' if cohen_d < 0.5 else
                                'a moderate impact' if cohen_d < 0.8 else
                                'a substantial impact')
                detect = "it truly detects a difference in mean" if power_of_test >= 0.8 else \
                "it might be incorrectly arguing that difference in mean is significant"

                print(f"The experiment led to a {significance} difference in mean.")
                print(f"The Bayesian Factor {supports}.")
                print(f"It shows that the experiment had practically {effect_size} on  change in mean.")
                print(f"The statistical power of test indicates {detect}.\n")

        return unpaired_ttest
    
    def get_statistics_before_after_rpt(self, df: pd.DataFrame, rpt_cycle: int = 201, neighbors_cycle: int = 3,
        window: int = 5, sigma_multiplier: float = 0.1, avoid_close_cycle: int = 0) -> pd.DataFrame:
        """
        Calculates statistics for cycles before and after the specified rpt_test_cycle.
        
        Parameters:
            df (pd.DataFrame): The input DataFrame.
            rpt_cycle (int): The cycle where the RPT is done.
            neighbors_cycle (int): Number of cycles before and after RPT.
            window (int): Rolling window size for outlier handling.
            sigma_multiplier (float): Sigma multiplier for outlier detection.
            avoid_close_cycle (int): Number of cycles to avoid immediately before and after rpt_test_cycle.
            
        Returns:
            pd.DataFrame: DataFrame with statistics for each cycle around the specified rpt_test_cycle.
        """
        # Identify cycles of interest
        cycles_before = list(range(rpt_cycle - 1 - neighbors_cycle - avoid_close_cycle, rpt_cycle - 1 - avoid_close_cycle))
        cycles_after = list(range(rpt_cycle + 1 + avoid_close_cycle, rpt_cycle + 1 + neighbors_cycle + avoid_close_cycle))
        cycles_of_interest = cycles_before + cycles_after

        # Get discharge energy and calculate normalized discharge energy
        df = self.get_discharge_energy(df, raw = True, disclaimer = False)
        df['norm_discharge_energy'] = df['adjusted_raw_discharge_energy'] / df.groupby(['doe_number', 'instance_number'])\
        ['adjusted_raw_discharge_energy'].transform(lambda x: x.iloc[1])

        # Clean data before and after RPT separately and keep rpt_cycle and rpt_cycle - 1 as it is
        df = df.groupby(['doe_number', 'instance_number']).apply(
            lambda group: pd.concat([
                group[group['cycle'] < rpt_cycle - 1].assign(norm_discharge_energy = lambda g: 
                self.outlier_handler(g['norm_discharge_energy'], window, sigma_multiplier)),
                group[group['cycle'].between(rpt_cycle - 1, rpt_cycle)],
                group[group['cycle'] > rpt_cycle].assign(norm_discharge_energy = lambda g: 
                self.outlier_handler(g['norm_discharge_energy'], window, sigma_multiplier))
            ])
        ).reset_index(drop = True)

        statistics = df[df['cycle'].isin(cycles_of_interest)].groupby(['doe_number', 'cycle']).\
        apply(lambda group: pd.Series(
              {'min': group['norm_discharge_energy'].min(),
                'avg':group['norm_discharge_energy'].mean(),
                'max': group['norm_discharge_energy'].max(),
                'std': group['norm_discharge_energy'].std(ddof = 1),
                'cv_pc': (group['norm_discharge_energy'].std(ddof = 1) * 100)/ \
                group['norm_discharge_energy'].mean() if group['norm_discharge_energy'].mean() > 0 else float('inf'),
                'sample_size': int(len(group))
            })).reset_index()
        # Perform ttest to see if RPT test had significant increase in mean discharge energy
        try:
            energy_before_rpt = df[df['cycle'].isin(cycles_before)]['norm_discharge_energy']
            energy_after_rpt = df[df['cycle'].isin(cycles_after)]['norm_discharge_energy']
            
            if len(energy_before_rpt) > 1 and len(energy_after_rpt) > 1:
                print(f"Sample size before RPT: {len(energy_before_rpt)}; Sample size after RPT: {len(energy_after_rpt)}.")
                print("Since correction = True, the Welch–Satterthwaite equation is used to approximate the adjusted degrees of freedom.")
                ttest = self.ttest(energy_after_rpt, energy_before_rpt, alternative = \
                        'greater').reset_index(drop = True)
                print(ttest)
        except Exception as e:
            print(e)
        
        return statistics
    
    def get_energy_cycle_gain(self, df: pd.DataFrame, rpt_cycle: int = 201, neighbors_cycle: int = 5, 
        avoid_close_cycle: int = 0, window: int = 5, sigma_multiplier: float = 0.1, show_plot: bool = True) -> pd.DataFrame:
        """
        Calculate energy gains and statistics for each instance around a specified RPT cycle.

        This function cleans the energy data by removing outliers and then calculates statistical 
        measures before and after a specified RPT cycle for each instance. It returns a DataFrame 
        containing the computed statistics and energy gains.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing 'doe_number', 'instance_number', 'cycle', 
                            'adjusted_raw_discharge_energy' columns.
            rpt_cycle (int): The reference cycle number where the RPT test is conducted. Default is 201.
            neighbors_cycle (int): The number of cycles to consider before and after the RPT cycle for 
                                statistical calculations. Default is 5.
            avoid_close_cycle (int): The number of cycles to avoid around the RPT cycle when calculating 
                                    statistics. Default is 0.
            window (int): The window size for the rolling function used in outlier detection. Default is 5.
            sigma_multiplier (float): The multiplier for the standard deviation to determine outliers. 
                                    Default is 0.1.

        Returns:
            pd.DataFrame: A DataFrame containing the computed statistics and energy gains for each instance.
        """

        # Identify cycles of interest
        cycles_before = list(range(rpt_cycle - 1 - neighbors_cycle - avoid_close_cycle, rpt_cycle - 1 - avoid_close_cycle))
        cycles_after = list(range(rpt_cycle + 1 + avoid_close_cycle, rpt_cycle + 1 + neighbors_cycle + avoid_close_cycle))

        # Get discharge energy and calculate normalized discharge energy
        df = self.get_discharge_energy(df, raw = True)
        df['norm_discharge_energy'] = df['adjusted_raw_discharge_energy'] / df.groupby(['doe_number', 'instance_number'])\
        ['adjusted_raw_discharge_energy'].transform(lambda x: x.iloc[1])

        # Clean data before and after RPT separately and keep rpt_cycle and rpt_cycle - 1 as it is
        df = df.groupby(['doe_number', 'instance_number']).apply(
            lambda group: pd.concat([
                group[group['cycle'] < rpt_cycle - 1].assign(norm_discharge_energy = lambda g: 
                self.outlier_handler(g['norm_discharge_energy'], window, sigma_multiplier)),
                group[group['cycle'].between(rpt_cycle - 1, rpt_cycle)],
                group[group['cycle'] > rpt_cycle].assign(norm_discharge_energy = lambda g: 
                self.outlier_handler(g['norm_discharge_energy'], window, sigma_multiplier))
            ])
        ).reset_index(drop = True)

        # Prepare list to collect results
        results = []
        
        # Group by doe_number and instance_number
        for (doe_number, instance_number), group in df.groupby(['doe_number', 'instance_number']):
            # Calculate statistics before and after
            stats_before = group.loc[group['cycle'].isin(cycles_before), 'norm_discharge_energy']
            stats_after = group.loc[group['cycle'].isin(cycles_after), 'norm_discharge_energy']
            
            avg_energy_before = stats_before.mean()
            median_energy_before = stats_before.median()
            std_energy_before = stats_before.std()
            avg_energy_after = stats_after.mean()
            median_energy_after = stats_after.median()
            std_energy_after = stats_after.std()

            # Gain in energy after the RPT test
            gain_in_energy = median_energy_after - median_energy_before

            # Gain in cycle
            if not stats_after.empty:
                closest_cycle_index = (group.loc[group['cycle'] > min(cycles_after), 
                        'norm_discharge_energy'] - avg_energy_before).abs().idxmin()
                close_cycle = group.at[closest_cycle_index, 'cycle']
                gain_in_cycle = close_cycle - max(cycles_before)
            else:
                gain_in_cycle = np.nan

            # Append the results
            results.append({
                'doe_number': doe_number,
                'instance_number': instance_number,
                'rpt_cycle': rpt_cycle,
                'avg_energy_before': avg_energy_before,
                'median_energy_before': median_energy_before,
                'std_energy_before': std_energy_before,
                'avg_energy_after': avg_energy_after,
                'median_energy_after': median_energy_after,
                'std_energy_after': std_energy_after,
                'gain_in_energy': gain_in_energy,
                'gain_in_cycle': gain_in_cycle
            })
        results = pd.DataFrame(results)
        # Introduce a gain in normalized energy as percentage
        results['gain_in_energy_pc'] = (results['gain_in_energy'] * 100)/results['median_energy_before']
        print(f"The Pearson's correlation coefficient between gain in energy and gain in cycle: {results['gain_in_energy_pc'].corr(results['gain_in_cycle'])}")
        if show_plot:
            data = results[results['gain_in_energy'] > 0].reset_index(drop = True)
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (20, 8))
            sns.scatterplot(data = data, x = 'gain_in_energy_pc', y = 'gain_in_cycle', color = 'blue', s = 100, ax = ax1)
            sns.histplot(data = data, x = 'gain_in_energy_pc', stat = 'probability', color = 'orange', ax = ax2)
            sns.histplot(data = data, x = 'gain_in_cycle', stat = 'probability', color = 'green', ax = ax3)
            ax1.yaxis.set_major_locator(MaxNLocator(integer = True))
            ax1.set_xlabel('$\Delta$ Normalized Discharge Energy [%]')
            ax2.set_xlabel('$\Delta$ Normalized Discharge Energy [%]')
            ax1.set_ylabel('$\Delta$ Cycle Number')
            ax3.set_xlabel('$\Delta$ Cycle Number')
            plt.tight_layout()
            plt.suptitle(f"{results.at[0, 'doe_number']}: Gain in Normalized Discharge Energy and Cycle Number after RPT Test for {len(results)} Packs")
            plt.subplots_adjust(top = 0.93)
            plt.show()
        return results
    
    def get_dcir(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get DCIR data.

        Parameters:
            df (pd.DataFrame): Input cycloid DataFrame with dcir column.

        Returns:
            pd.DataFrame: DataFrame with DCIR data.
        """
        dcir_df = df.loc[df['dcir'].notnull(), ['doe_number', 'instance_number', 'cycle', 'dcir']].reset_index(drop = True)
        dcir_df['dcir'] = (dcir_df['dcir'] * 1000).round(1)
        return dcir_df
    
    def plot_dcir(self, df: pd.DataFrame, hue: str = 'instance_number', label_title: str = 'Instance Number', ax = None) -> None:
        """
        Plot DCIR data.

        Parameters:
            df (pd.DataFrame): Input DataFrame from get_dcir() method.
            hue (str, optional): The column name to use for coloring the lines. Default is 'instance_number'.
            label_title (str, optional): Title for the legend. Default is 'Instance Number'.
            ax (matplotlib.axes.Axes, optional): Axes object to plot on. If not provided, a new figure will be created.
        """
        dcir = self.get_dcir(df)
        unique_hues = df[hue].nunique()

        if unique_hues < 8:
            palette = ['red', 'blue', 'black', 'magenta', 'orange', 'gray', 'green', 'navy'][:unique_hues]
        else:
            palette = self.sns.color_palette('colorblind')[:unique_hues]

        if ax is None:
            _, ax = plt.subplots(figsize = (12, 8))

        sns.lineplot(data = dcir, x = 'cycle', y = 'dcir', hue = hue, palette = palette, ax = ax)
        ax.set_ylabel("DCIR [m$\Omega$]")
        ax.set_xlabel("Cycle Number")
        ax.legend(labelcolor = 'linecolor', title = label_title)

    def plot_mean_voltage(self, df: pd.DataFrame, hue: str = 'instance_number', 
                charging_state_name: str = 'CHARGE', label_title: str = 'Instance Number', 
                raw_data = True, window: int = 5, sigma_multiplier: float = 0.2, ax = None) -> None:
        """
        Plot cycloid pack/cell mean voltage data.

        Parameters:
            df (pd.DataFrame): Input DataFrame.
            hue (str, optional): The column name to use for coloring the lines. Default is 'instance_number'.
            charging_state_name (str, optional): Substate for which to plot mean voltage. Default is 'CHARGE'.
            label_title (str, optional): Title for the legend. Default is 'Instance Number'.
            ax (matplotlib.axes.Axes, optional): Axes object to plot on. If not provided, a new figure will be created.
        """
        unique_hues = df[hue].nunique()

        if unique_hues < 8:
            palette = ['red', 'blue', 'black', 'magenta', 'orange', 'gray', 'green', 'navy'][:unique_hues]
        else:
            palette = self.sns.color_palette('colorblind')[:unique_hues]
        
        df = df[df['cycling_substate'] == charging_state_name].reset_index(drop = True)

        if (not raw_data) and (df['instance_number'].nunique() > 1):
            df['mean_mean_voltage'] = df.groupby('instance_number').apply(lambda group: 
            self.outlier_handler(group['mean_mean_voltage'], window = window, 
            sigma_multiplier = sigma_multiplier)).values

        elif not raw_data:
             df['mean_mean_voltage'] = self.outlier_handler(df['mean_mean_voltage'], window = window, 
                sigma_multiplier = sigma_multiplier).values
             
        else:
            print("The mean_mean_voltage plotted against cycle number without cleaning outliers", end = ' ')
            print("use raw_data = False to clean the outliers")

        if ax is None:
            _, ax = plt.subplots(figsize = (10, 8))

        sns.lineplot(data = df, x = 'cycle', y = 'mean_mean_voltage', hue = hue, palette = palette, ax = ax)
        ax.set_ylabel("Average Charge Voltage [V]")
        ax.set_xlabel("Cycle Number")
        ax.legend(labelcolor = 'linecolor', title = label_title)

    def plot_rover_metrics(self, df: pd.DataFrame, x:str = 'elapsed_minutes', charging_state_name: str = 'CHARGE',
     hue: str = None, suptitle: str = None, label_title: str = None, legend: bool = True) -> None:
        """
        Plot up to 6 metrics in a subplot based on user input.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the data to plot.
            charging_state_name (str, optional): The charging state to plot. Default is CHARGE.
            hue (str, optional): The column name to use for coloring the lines. Default is None.
            suptitle (str, optional): Title for the entire plot. Default is None.
            label_title (str, optional): Title for the legend labels. Default is None.
            legend (bool, optional): Legend ofr each subplot. Default is True.
        """
        # get the select charging state data
        if charging_state_name in df['charging_state_name'].unique():
            df = df[df['charging_state_name'] == charging_state_name].reset_index(drop = True)
            df['doe_instance'] = df['doe'].astype(str) + '_' + df['instance'].astype(str)
        else:
            print(f"There is no charging state {charging_state_name} in the DataFrame.")

        # Create a palette for hue
        if hue is not None:
            unique_hues = df[hue].nunique()
            if unique_hues < 8:
                palette = ['red', 'blue', 'black', 'magenta', 'orange', 'gray', 'green', 'navy'][:unique_hues]
            else:
                palette = self.sns.color_palette('colorblind')[:unique_hues]
        
        # Lets make sure the current is always positive
        df['avg_current_a'] = df['avg_current_a'].abs()

        # List of available metrics
        available_metrics = ['avg_current_a', 'avg_voltage_v',
                            'avail_battery_capacity_ah', 'battery_temperature_deg_c',
                            'min_current_a', 'max_current_a', 'min_voltage_v',
                            'max_voltage_v', 'current_a', 'voltage_v']

        metrics_label_dict = {
                            'avg_current_a': 'Average Current [A]',
                            'avg_voltage_v': 'Average Voltage [V]',
                            'avail_battery_capacity_ah': 'Capacity [Ah]',
                            'battery_temperature_deg_c': 'Battery Temperature [$^\circ$C]',
                            'min_voltage_v': 'Min Voltage [V]',
                            'max_voltage_v': 'Max Voltage [V]',
                            'max_current_a': 'Max Current [A]',
                            'min_current_a': 'Min Current [A]',
                            'current_a': 'Current [A]',
                            'voltage_v': 'Voltage [V]',
                            'elapsed_minutes': 'Time [minutes]'
                            }

        # Prompt user to select metrics
        print("Available metrics:")
        for i, metric in enumerate(available_metrics, start = 1):
            print(f"{i}.{metric}")

        selected_metrics = []
        while True:
            selection = input("Enter the number of the metric to plot (0 to finish): ")
            if selection == '0':
                break
            elif selection.isdigit() and 1 <= int(selection) <= len(available_metrics):
                selected_metrics.append(available_metrics[int(selection) - 1])
            else:
                print("Invalid input. Please enter a number between 1 and", len(available_metrics))

        # Plot selected metrics
        n = min(len(selected_metrics), 6)  # Maximum of 6 metrics in a subplot

        if n == 0:
            print("No metrics selected.")
            return

        rows = (n + 1) // 2  # Number of rows in the subplot

        if n == 1:
            fig, ax = plt.subplots(figsize = (10, 8))
            ax = [ax]  # Make ax a list to keep the indexing consistent
        else:
            fig, ax = plt.subplots(rows, 2, figsize = (4 * n, 3 * n)) if n > 2 else  plt.subplots(rows, 2, figsize = (16, 8))
            ax = ax.flatten()

        for i, metric in enumerate(selected_metrics):
            if hue is not None:
                sns.lineplot(data = df, x = x, y = metric, hue = hue, palette = palette, ax = ax[i])
            else:
                sns.lineplot(data = df, x = x, y = metric, ax = ax[i])
            ax[i].set_xlabel(metrics_label_dict.get(x, x))
            ax[i].set_ylabel(metrics_label_dict.get(metric, metric))
            if legend:
                ax[i].legend(labelcolor = 'linecolor', title = label_title) if label_title is not None \
                    else ax[i].legend(labelcolor='linecolor')
            else:
                ax[i].legend().set_visible(False)

        # Hide empty subplots if any
        for i in range(n, len(ax)):
            fig.delaxes(ax[i])

        if suptitle is not None:
            plt.suptitle(suptitle)

        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        plt.show()


    def read_data(self, data_file_path: str) -> pd.DataFrame:
        if data_file_path.endswith('.csv'):
            df = pd.read_csv(data_file_path)
        elif data_file_path.endswith('.xlsx'):
            df = pd.read_excel(data_file_path)
        else:
            print("The data file extension should be either .csv or .xlsx.")

        df.rename(columns = {'doe_number': 'doe', 'instance_number': 
        'instance', 'cycle': 'cycle_num', 'cycling_substate': 'charging_state_name'}, inplace = True)

        # Lets look at unique does we have
        print(f"The unique does in the data: {list(df.doe.unique())}\n")

        # Lets look at unique instances we have
        print(f"The unique instances in the data: {list(df.instance.unique())}\n")

        # Lets look at unique cycles we have
        if len(df.cycle_num.unique()) <= 6:
            print(f"The list of cycles in the data: {list(df.cycle_num.unique())}")
        else:
            print(f"The range of cycles in the data: {df.cycle_num.min(), df.cycle_num.max()}")

        return df


    def filter_by_doe(self, df: pd.DataFrame, doe_list: list) -> pd.DataFrame:
        """
        Filter the DataFrame by the provided doe_value.

        Parameters:
            doe_value (list): The list of DOE to filter.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        return df[df['doe'].isin(doe_list)].reset_index(drop = True)
    
    def filter_by_instance(self, df: pd.DataFrame, instance_list: list) -> pd.DataFrame:
        """
        Filter the DataFrame by the provided doe_value.

        Parameters:
            doe_value (list): The list of DOE to filter.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        return df[df['instance'].isin(instance_list)].reset_index(drop = True)
    
    def filter_by_cycle(self, df: pd.DataFrame, cycle_list: list) -> pd.DataFrame:
        """
        Filter the DataFrame by the provided doe_value.

        Parameters:
            doe_value (list): The list of DOE to filter.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        return df[df['cycle_num'].isin(cycle_list)].reset_index(drop = True)
    
    def filter_by_doe_instance(self, df: pd.DataFrame, doe_instance_tuple: list[tuple[Union[int, str], Union[int, str]]]) -> pd.DataFrame:
        """
        Filter the DataFrame by the provided doe_instance_tuple.

        Parameters:
            doe_instance_values (list of tuples): List of tuples containing (doe, instance) values to filter.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        filtered_dfs = []
        for doe, instance in doe_instance_tuple:
            filtered_df = df[(df['doe'] == doe) & (df['instance'] == instance)]
            filtered_dfs.append(filtered_df)
        return pd.concat(filtered_dfs, ignore_index = True)
    
    def get_cells_voltage_std(self, df: pd.DataFrame, window: int = 5, 
        sigma_multiplier : Union[int,float] = 0.1) -> dict[pd.DataFrame]:
        """
        Calculates the standard deviation (with ddof = 1) for 5 cells in a pack for each cycle
        and cycling substate. The cycloid DataFrame for each cycling substate with new added column
        standard deviation of cells voltage is added to DataFrame of each cycling substate. it ruturns
        the dictionary of cycling substate as key and its corresponding DatFrame as value.

        Parameters:
            df : Input cycloid DataFrame with cells min and max voltage columns.

        Returns:
            dict : Dictionary with key as each cycling substate and value as DataFrame of correponding
            substate with standard deviation of cells voltage added for each cycle and substate. 
        """
        req_cols = ['doe_number', 'instance_number', 'cycling_substate', 'cycle']
        voltage_columns = {
            'CHARGE': [f'max_cell{i}_voltage_v' for i in range(1, 6)],
            'CHARGE REST': [f'min_cell{i}_voltage_v' for i in range(1, 6)],
            'DISCHARGE':  [f'min_cell{i}_voltage_v' for i in range(1, 6)],
            'DISCHARGE REST':  [f'max_cell{i}_voltage_v' for i in range(1, 6)]
        }
        state_dict = {}
        for state, cols in voltage_columns.items():
            data = df[df['cycling_substate'] == state][req_cols + cols].dropna()
            for col in cols:
                 data[col] = data.groupby(['doe_number', 'instance_number'])[col]\
                 .transform(lambda series: self.outlier_handler(series, window = window, sigma_multiplier = sigma_multiplier))
            data['std_mv'] = round((data[cols].std(axis = 1, ddof = 1)) * 1000)
            state_dict[state] = data.sort_values(by = ['doe_number', 'instance_number', 'cycle']).reset_index(drop = True)

        return state_dict
    
    def plot_cells_voltage_std(self, df: pd.DataFrame) -> None:
        """
       Takes input as DataFrame and uses the output DataFrame of get_cells_voltage_std() method. 
       It displays a plot with mean and 95% C.I. of standard deviation of 5 cells voltage versus 
       cycle number for each cycling substate.

        Parameters:
            df_dict : Input dictionary.

        Returns:
            None: Displays the plot. 
        """
        df_dict = self.get_cells_voltage_std(df)
        doe = df_dict['CHARGE'].doe_number.iloc[0]
        num_packs = len(df_dict['CHARGE'].instance_number.unique())
        
        # set common  y lim for across all DoEs if comparing multiple DoEs
        ylim_dict = {
             'CHARGE': [2, 24],
            'CHARGE REST': [2, 14],
            'DISCHARGE': [10, 250],
            'DISCHARGE REST': [10, 32]
        }
        sns.set_style("whitegrid")
        sns.set_palette('colorblind')
        fig, ax = plt.subplots(2, 2, figsize = (16, 14))
        ax = ax.flatten()
        for i, (state, data) in enumerate(df_dict.items()):
            sns.lineplot(data = data, x = 'cycle', y = 'std_mv', color = 'red', ax = ax[i])
            ax[i].set_ylim(ylim_dict.get(state))
            # ax[i].set_xticks(np.arange(0, 351, 50))
            ax[i].yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
            ax[i].set_ylabel(f"Std. Dev. of {state.title()} Voltage [mV]")
            ax[i].set_xlabel('Cycle Number')
        plt.tight_layout()
        plt.suptitle(f"{doe} " + r"$\bar{x}$ and $95\%~C.I.$ of Std. Dev. of 5 Cells per Cycle for " + f"{num_packs} Packs", fontweight = 'bold')
        plt.subplots_adjust(top = 0.95)
        plt.show()

    def plot_cells_voltage(self, df: pd.DataFrame, instance_number: Union[int, str] = None, 
        raw_data = True, window: int = 5, sigma_multiplier: Union[float, int] = 0.1):
        """
        Plot voltage data for different cycling states.
        
        Parameters:
            df (DataFrame): Input cycloid DataFrame containing voltage data.
            instance_number (int or str): Instance number to plot (default is None).
            raw_data (bool): If True, plots raw data. If False, plots clean data.
            window (int): Window size for rolling median and standard deviation (default is 5).
            sigma_multiplier (float): Sigma multiplier for outlier detection (default is 0.1).
        
        Returns:
            None: Displays a 2 * 2 plot with cells voltage for each cycling substate.
        """
        
        # Get the doe, doe should be unique
        doe = df.doe_number.iloc[0]

        # Filter DataFrame by instance number
        if instance_number is None:
            instance_sample = df.instance_number.unique()
            instance_number = np.random.choice(instance_sample, 1)[0] if len(instance_sample) > 1 else instance_sample[0]

        df_instance = df[df.instance_number == instance_number].reset_index(drop = True)
        
        # Define voltage column mappings for different states
        voltage_columns = {
            'CHARGE': 'max_cell{}_voltage_v',
            'CHARGE REST': 'min_cell{}_voltage_v',
            'DISCHARGE': 'min_cell{}_voltage_v',
            'DISCHARGE REST': 'max_cell{}_voltage_v'
        }
        
        # Define color palette
        palette = ['blue', 'black', 'orange', 'green', 'red']

        voltage_data = {}
        for state in voltage_columns.keys():
            col = [voltage_columns[state].format(j) for j in range(1, 6)]
            h = df_instance[df_instance['cycling_substate'] == state][['cycle'] + col].dropna().reset_index(drop = True)
            x = pd.melt(h, id_vars = ['cycle'], value_vars = col, var_name = 'cell', value_name = 'voltage')
            x['cell'] = x['cell'].str.extract(r'(cell\d)').squeeze()
            voltage_data[state] = x
        

        if not raw_data:
            for state in voltage_data.keys():
                voltage_data[state]['voltage'] = voltage_data[state].groupby(['cell'])['voltage'].transform(lambda series: self.outlier_handler(series, 
                window = window, sigma_multiplier = sigma_multiplier))
        else:
            print("You are using raw data for your plot, for clean data use 'raw_data = False' parameter.")
        
        # # Create subplots
        fig, ax = plt.subplots(2, 2, figsize = (16, 14))
        ax = ax.flatten()

        # Define color palette
        palette = ['blue', 'black', 'orange', 'green', 'red']

        # Plot voltage data for each state
        for i, (state_name, data) in enumerate(voltage_data.items()):
            # Plot voltage data
            sns.lineplot(data = data, x = 'cycle', y = 'voltage', hue = 'cell', palette = palette, ax = ax[i])
            ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax[i].set_xlabel("Cycle Number")
            ax[i].set_ylabel(f"{voltage_columns.get(state_name).split('_')[0].title()} {state_name.title()} Cells Voltage [V]")
            ax[i].legend(labelcolor='linecolor')
        
        plt.suptitle(f"{doe}_{instance_number} Cycling Substates Cells Voltage vs. Cycle")
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.show()

    def get_polarization_voltage(self, df: pd.DataFrame):
        """
        Within each cycling_substate, to do the max - min of the CELL/PACK voltage.
        """
        data = df[df['cycling_substate'].isin(['CHARGE REST', 'DISCHARGE REST'])].reset_index(drop = True)
        return data.groupby(['doe_number', 'instance_number', 'cycle', 'cycling_substate']).apply(
        lambda group: pd.Series({**{
            f'cell{i}_delta_voltage': group[f'max_cell{i}_voltage_v'].values[0] - 
            group[f'min_cell{i}_voltage_v'].values[0] for i in range(1, 6)},
            'pack_delta_voltage': group['max_rest_voltage_v'].values[0] - group['min_rest_voltage_v'].values[0]})
            ).reset_index()


    def plot_polarization_voltage(self, df:pd.DataFrame, window: int = 5, sigma_multiplier:Union[int, float] = 0.1,
         raw_data: bool = True, legend: bool = True):
            
            df = self.get_polarization_voltage(df)

            cols_to_plot = ['pack_delta_voltage'] + [f'cell{i}_delta_voltage' for i in range(1, 6)]
            doe = df['doe_number'].iloc[0]
            
            # Create a palette for hue
            unique_hues = df['instance_number'].nunique()
            if unique_hues < 8:
                palette = ['red', 'blue', 'black', 'magenta', 'orange', 'gray', 'green', 'navy'][:unique_hues]
            else:
                palette = self.sns.color_palette('colorblind')[:unique_hues]

            if not raw_data:
                for col in cols_to_plot:
                    df[col] = df.groupby(['doe_number', 'instance_number', 'cycling_substate'])[col].\
                    transform(lambda series: self.outlier_handler(series, window = window, sigma_multiplier = sigma_multiplier))
            else:
                print("You are using raw data for your plot, for clean data use 'raw_data = False' parameter.")
            
            for state in ['CHARGE REST', 'DISCHARGE REST']:
                data = df[df['cycling_substate'] == state].reset_index(drop = True)

                fig, ax = plt.subplots(2, 3, figsize = (20, 16), sharex = True)
                ax = ax.flatten()

                sns.lineplot(data = data, x = 'cycle', y = cols_to_plot[0], hue = 'instance_number', palette = palette, ax = ax[0])
                ax[0].set_xlabel("Cycle Number")
                ax[0].set_ylabel("Pack Delta Voltage [V]")
                ax[0].legend(labelcolor = 'linecolor')
                if legend:
                        ax[0].legend(labelcolor = 'linecolor', title = 'Instance')
                else:
                        ax[0].legend().set_visible(False)


                # Initialize variables to store min and max global y-axis limits
                global_min_y = float('inf')
                global_max_y = float('-inf')

                for i, col in enumerate(cols_to_plot[1:], start = 1):
                    sns.lineplot(data = data, x = 'cycle', y = col, hue = 'instance_number', palette = palette, ax = ax[i])
                    ylabel = ' '.join(col.split('_')).title() + ' [V]'
                    ax[i].set_ylabel(ylabel)
                    ax[i].set_xlabel("Cycle Number")
                    if legend:
                        ax[i].legend(labelcolor = 'linecolor', title = 'Instance')
                    else:
                        ax[i].legend().set_visible(False)
                    min_y, max_y = ax[i].get_ylim()
                    if min_y < global_min_y:
                        global_min_y = min_y
                    if max_y > global_max_y:
                        global_max_y = max_y

                for i in range(1, len(cols_to_plot[1:]) + 1):
                    ax[i].set_ylim([global_min_y, global_max_y])
                
                plt.suptitle(f"{doe} {state.title()} Polarization Voltage for Pack and Individual Cells")
                
                plt.tight_layout()

    def plot_temperature(self, df: pd.DataFrame, x: str = 'cycle', y: str = 'max_temp', hue: str = 'instance_number',
        clean_first_n_cycles: int = 5, window: int = 7, sigma_multiplier: Union[int, float] = 0.5, sharex: bool = True) -> None:
        # Lets get the doe for suptitle, if more than 1 doe, will have to modify manually
        doe = df.iloc[0]['doe_number'] if df['doe_number'].nunique() == 1 else ''
        # Define hue based on what column to use for hue: instance_number, doe_number, doe_instance
        unique_hues = df[hue].nunique()

        if unique_hues < 8:
            palette = ['red', 'blue', 'black', 'magenta', 'orange', 'gray', 'green', 'navy'][:unique_hues]
        else:
            palette = sns.color_palette('colorblind')[:unique_hues]

        # Create subplots
        fig, ax = plt.subplots(2, 2, figsize = (18, 14), sharex = sharex)
        ax = ax.flatten()
        # If you need to set up a limit for temperature for each state change here
        # temp_limits = {
        #                 'CHARGE': [34, 56],
        #                 'CHARGE REST': [34, 55],
        #                 'DISCHARGE': [53, 75],
        #                 'DISCHARGE REST': [55, 77]     
        #               }

        # Iterate through each charging state
        for i, state in enumerate(['CHARGE','CHARGE REST', 'DISCHARGE', 'DISCHARGE REST']): 
                # Filter DataFrame for the current state and cell index
                df_state = df[df['cycling_substate'] == state].dropna(subset = [y]).reset_index(drop = True)
                # First few cycles might have a spike in temperature, clean them
                md = df_state.loc[df_state[x] < clean_first_n_cycles, y].median()
                df_state.loc[df_state[x] < clean_first_n_cycles, y] = md
                df_state[y] = df_state.groupby(['doe_number', 'instance_number'])[y].transform(lambda group: 
                              self.outlier_handler(group, window = window, sigma_multiplier = sigma_multiplier))
                # Plot temperature data for each cycling substate
                sns.lineplot(data = df_state, x = x, y = y, hue = hue, palette = palette, ax = ax[i])
                ax[i].set_xlabel("Cycle Number") if x == 'cycle' else ax[i].set_xlabel(x)
                ax[i].set_ylabel(f"{state.title()} {y.split('_')[0].title()} Pack Temp [$^\circ$C]")
                # Set the y axis ticks as integers
                ax[i].yaxis.set_major_locator(MaxNLocator(integer = True))
                # If we need a common x and y lim, uncomment here
                # ax[i].set_ylim(temp_limits.get(state))
                ax[i].legend(labelcolor = 'linecolor', title = ' '.join([string.title() for string in hue.split('_')]))
        plt.suptitle(f"{doe} Battery Temperature vs. Cycle for all Cycling Substates")
        plt.tight_layout()
        plt.subplots_adjust(top = 0.95)

        plt.show()