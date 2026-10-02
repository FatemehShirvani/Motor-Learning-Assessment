"""Original motor-learning analysis exported from the sanitized notebook.

This file contains analysis code only. The confidential study inputs and
saved notebook outputs are intentionally excluded from the repository.
"""

# %% [markdown]
# # Motor-learning analysis notebook
#
# > Public code-only copy. All saved outputs and attachments were removed because the original study data are confidential. The notebook still references the expected private data-directory structure, but no participant data are included in this repository.
#

# %%
from google.colab import drive
drive.mount('/content/drive')

import scipy.io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Path to .mat files and the Excel file
mat_files_path = '/content/drive/MyDrive/project/Behavioral'
excel_file_path = '/content/drive/MyDrive/project/Movement Time.xlsx'

# Load the Excel file
movement_time_df = pd.read_excel(excel_file_path)

# Get a list of all .mat files in the directory
mat_files = [f for f in os.listdir(mat_files_path) if f.endswith('.mat')]

# Sort the list of files based on 'm' (session number) and then 'n' (subject number)
mat_files.sort(key=lambda x: (int(x.split('_S')[2].split('.')[0]), int(x.split('_S')[1])))

# Create a dictionary to store the data
mat_data = {}

# Loop through the list and read each .mat file
for mat_file in mat_files:
    full_path = os.path.join(mat_files_path, mat_file)
    mat_data[mat_file] = scipy.io.loadmat(full_path)



# %%
def extract_velocity(name, n, m, i):
    """
    Extract velocities v_x, v_y, and v_xy from the .mat file based on the given parameters,
    after normalizing the x and y positions.
    """
    for file_name, data in mat_data.items():
        if f"{name}_S{n}_S{m}" in file_name:
            for key, value in data.items():
                if isinstance(value, np.ndarray) and value.dtype == object:
                    main_array = value[0]  # Access the main array
                    if i <= len(main_array) and i > 0:
                        nested_array = main_array[i-1]
                        # Extract x and y positions
                        x_positions = nested_array[:, 0]
                        y_positions = nested_array[:, 1]

                        # Normalize x and y positions
                        x_min, x_max = np.min(x_positions), np.max(x_positions)
                        y_min, y_max = np.min(y_positions), np.max(y_positions)

                        x_normalized =  (x_positions - x_min) / (x_max - x_min)
                        y_normalized =  (y_positions - y_min) / (y_max - y_min)

                        # Find the corresponding row in the Excel file
                        row = movement_time_df[(movement_time_df['Subject'] == n) &
                                               (movement_time_df['Session'] == f'T{m}') &
                                               (movement_time_df['Shape'] == name) &
                                               (movement_time_df['Trial'] == i)]
                        if not row.empty:
                            time_value = row['Time'].values[0]
                            delta_t = time_value / len(x_normalized)

                            # Calculate velocities
                            """ v_x = (x_normalized[1:] - x_normalized[:-1]) / delta_t
                            v_y = (y_normalized[1:] - y_normalized[:-1]) / delta_t"""
                            # Time step (assuming uniform spacing)
                            h = delta_t

                            # Calculate velocities using 5-point stencil method
                            v_x = np.zeros_like(x_normalized)
                            v_y = np.zeros_like(y_normalized)

                            # Preallocate velocity array

                            v_x[0] = (x_normalized[1] - x_normalized[0]) / h
                            v_x[-1] = (x_normalized[-1] - x_normalized[-2]) / h
                            v_y[0] = (y_normalized[1] - y_normalized[0]) / h
                            v_y[-1] = (y_normalized[-1] - y_normalized[-2]) /h

                            # Apply 5-point stencil method for interior points (from index 2 to index length(x)-3)
                            for i in range(2, len(x_normalized) - 2):
                                v_x[i] = (-x_normalized[i+2] + 8*x_normalized[i+1] - 8*x_normalized[i-1] + x_normalized[i-2]) / (12*h)
                                v_y[i] = (-y_normalized[i+2] + 8*y_normalized[i+1] - 8*y_normalized[i-1] + y_normalized[i-2]) / (12*h)

                            # Use central difference for the second and second-to-last points
                            v_x[1] = (x_normalized[2] - x_normalized[0]) / (2*h)
                            v_x[-2] = (x_normalized[-1] - x_normalized[-3]) / (2*h)
                            v_y[1] = (y_normalized[2] - y_normalized[0]) / (2*h)
                            v_y[-2] = (y_normalized[-1] - y_normalized[-3]) / (2*h)

                            # Use forward difference for the first point


                            # Use backward difference for the last point
                            v_xy = np.sqrt(v_x**2 + v_y**2)

                            # Add NaN for the last row to match lengths
                            """v_x = np.append(v_x, np.nan)
                            v_y = np.append(v_y, np.nan)
                            v_xy = np.append(v_xy, np.nan)"""

                            # Time values starting from 0 and incrementing by delta_t
                            time_values = np.arange(0, len(x_normalized) * delta_t, delta_t)
                            return x_normalized, y_normalized, v_x, v_y, v_xy, time_values
                        else:
                            print(f"No matching row found in the Excel file for {name}, S{n}, S{m}, Trial {i}")
                            return None
                    else:
                        print(f"Array index {i} out of bounds for file {file_name}")
                        return None
    return None


def extract_velocity_previous(name, n, m, i):
    """
    Extract velocities v_x, v_y, and v_xy from the .mat file based on the given parameters,
    after normalizing the x and y positions.
    """
    # Ensure the 'len' column exists in movement_time_df
    if 'len' not in movement_time_df.columns:
        movement_time_df['len'] = np.nan

    for file_name, data in mat_data.items():
        if f"{name}_S{n}_S{m}" in file_name:
            for key, value in data.items():
                if isinstance(value, np.ndarray) and value.dtype == object:
                    main_array = value[0]  # Access the main array
                    if i <= len(main_array) and i > 0:
                        nested_array = main_array[i-1]
                        # Extract x and y positions
                        x_positions = nested_array[:, 0]
                        y_positions = nested_array[:, 1]

                        # Normalize x and y positions
                        x_min, x_max = np.min(x_positions), np.max(x_positions)
                        y_min, y_max = np.min(y_positions), np.max(y_positions)

                        x_normalized =  (x_positions - x_min) / (x_max - x_min)
                        y_normalized =  (y_positions - y_min) / (y_max - y_min)

                        # Find the corresponding row in the Excel file
                        row = movement_time_df[(movement_time_df['Subject'] == n) &
                                               (movement_time_df['Session'] == f'T{m}') &
                                               (movement_time_df['Shape'] == name) &
                                               (movement_time_df['Trial'] == i)]
                        if not row.empty:
                            time_value = row['Time'].values[0]
                            delta_t = time_value / len(x_normalized)

                            # Calculate velocities
                            v_x = (x_normalized[1:] - x_normalized[:-1]) / delta_t
                            v_y = (y_normalized[1:] - y_normalized[:-1]) / delta_t
                            v_xy = np.sqrt(v_x**2 + v_y**2)

                            # Add NaN for the last row to match lengths
                            v_x = np.append(v_x, np.nan)
                            v_y = np.append(v_y, np.nan)
                            v_xy = np.append(v_xy, np.nan)

                            # Time values starting from 0 and incrementing by delta_t
                            time_values = np.arange(0, len(x_normalized) * delta_t, delta_t)

                            # Update the 'len' column with the length of the sample
                            movement_time_df.loc[row.index, 'len'] = len(x_normalized)

                            # Save the updated dataframe (optional, depending on how you manage the file)

                            movement_time_df.to_csv('updated_movement_time.csv', index=False)

                            return x_normalized, y_normalized, v_x, v_y, v_xy, time_values
                        else:
                            print(f"No matching row found in the Excel file for {name}, S{n}, S{m}, Trial {i}")
                            return None
                    else:
                        print(f"Array index {i} out of bounds for file {file_name}")
                        return None
    return None


def smooth_data(data, window_length=51, polyorder=3):
    """
    Smooth the data using a Savitzky-Golay filter.
    """
    return savgol_filter(data, window_length=window_length, polyorder=polyorder, mode='nearest')


def remove_outliers(data, time_values, factor=3.9):
    """
    Replace outliers in the data with linearly interpolated values using the IQR method.
    """


    Q1 = np.nanpercentile(data, 25)
    Q3 = np.nanpercentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    outliers = np.where((data < lower_bound) | (data > upper_bound))[0]

    #for outlier in outliers:
        #print(f"Outlier found at time {time_values[outlier]:.2f}s: {data[outlier]:.2f}")

    # Create a pandas Series to use its interpolation method
    data_series = pd.Series(data)

    # Replace outliers with NaN
    data_series[outliers] = np.nan

    # Interpolate the NaN values linearly
    data_interpolated = data_series.interpolate(method='linear', limit_direction='both')

    #for outlier in outliers:
        #print(f"Interpolated value at time {time_values[outlier]:.2f}s: {data_interpolated[outlier]:.2f}")

    return data_interpolated.to_numpy()


def process_velocity(v_xy, time_values):
    """
    Smooth and remove outliers from the velocity data.
    """
    v_xy_processed = v_xy
    #v_xy_processed = smooth_data(v_xy_processed)
    v_xy_processed = remove_outliers(v_xy_processed, time_values)
    v_xy_processed = smooth_data(v_xy_processed)
    return v_xy_processed

def extract_acceleration(v_xy, time_values):
    """
    Calculate acceleration from velocity data.
    """
    if len(v_xy) != len(time_values):
        delta_t = np.diff(time_values[:-(len(time_values)-len(v_xy))])

    else:
        delta_t = np.diff(time_values)
    acceleration = np.diff(v_xy) / delta_t
    return acceleration


def process_acceleration(acceleration, time_values):
    """
    Smooth and remove outliers from the acceleration data.
    """
    acceleration_processed = acceleration
    #acceleration_processed = smooth_data(acceleration_processed)
    acceleration_processed = remove_outliers(acceleration_processed, time_values)
    acceleration_processed = smooth_data(acceleration_processed)
    return acceleration_processed


def extract_jerk(acceleration, time_values):
    """
    Calculate jerk from acceleration data.
    """
    if len(acceleration) != len(time_values):
        delta_t = np.diff(time_values[:-(len(time_values)-len(acceleration))])

    else:
        delta_t = np.diff(time_values)

    jerk = np.diff(acceleration) / delta_t
    return jerk


def process_jerk(jerk, time_values):
    """
    Smooth and remove outliers from the jerk data.
    """
    jerk_processed = jerk
    #jerk_processed = smooth_data(jerk_processed)
    jerk_processed = remove_outliers(jerk_processed, time_values)
    jerk_processed = smooth_data(jerk_processed)
    return jerk_processed


def plot_velocity(name, n, m, i):
    """
    Plot v_x, v_y, and v_xy velocities from the extracted data.
    """
    result = extract_velocity(name, n, m, i)
    if result is not None:
        x_positions, y_positions, v_x, v_y, v_xy, time_value = result
        v_xy_processed = process_velocity(v_xy, time_value)
        delta_t = time_value / len(x_positions)

        # Plot velocities for v_x
        """plt.figure()
        plt.plot(np.arange(len(v_x)) * delta_t, v_x, label='v_x', marker='o', linestyle='-')
        plt.title(f'v_x Velocity from {name}_S{n}_S{m}_T{i}')
        plt.xlabel('Time')
        plt.ylabel('Velocity (v_x)')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot velocities for v_y
        plt.figure()
        plt.plot(np.arange(len(v_y)) * delta_t, v_y, label='v_y', marker='o', linestyle='-')
        plt.title(f'v_y Velocity from {name}_S{n}_S{m}_T{i}')
        plt.xlabel('Time')
        plt.ylabel('Velocity (v_y)')
        plt.legend()
        plt.grid(True)
        plt.show()"""

        # Plot tangential velocities for v_xy
        plt.figure()
        plt.plot(np.arange(len(v_xy_processed)) * delta_t, v_xy_processed, label='v_xy', linestyle='-')
        plt.title(f'v_xy Tangential Velocity from {name}_S{n}_S{m}_T{i}')
        plt.xlabel('Time')
        plt.ylabel('Tangential Velocity (v_{xy})')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print(f"Failed to plot velocity for {name}_S{n}_S{m}_T{i}")

#plot_velocity("B", 12, 1, 1)



def plot_velocity_acceleration_jerk(name, n, m, i):
    """
    Plot velocity, acceleration, and jerk from the extracted data.
    """
    result = extract_velocity(name, n, m, i)
    if result is not None:
        x_positions, y_positions, v_x, v_y, v_xy, time_values = result
        v_xy_processed = process_velocity(v_xy, time_values)
        acceleration = extract_acceleration(v_xy_processed, time_values)
        acceleration_processed = process_acceleration(acceleration, time_values)
        jerk = extract_jerk(acceleration_processed, time_values)
        jerk_processed = process_jerk(jerk, time_values)

        # Adjust time values to match the length of each dataset
        time_values_vxy = time_values[:len(v_xy_processed)]
        time_values_acc = time_values[1:len(acceleration_processed) + 1]
        time_values_jerk = time_values[2:len(jerk_processed) + 2]

        plt.figure(figsize=(15, 10))

        # Plot v_xy
        plt.subplot(3, 1, 1)
        plt.plot(time_values_vxy, v_xy_processed, label='v_xy', linestyle='-')
        plt.title(f'{name}-S{n}-S{m}-T{i}-Velocity (v_xy)')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (v_xy)')
        plt.grid(True)
        plt.legend()

        # Plot acceleration
        plt.subplot(3, 1, 2)
        plt.plot(time_values_acc, acceleration_processed, label='Acceleration', linestyle='-')
        plt.title(f'{name}-S{n}-S{m}-T{i}- Acceleration')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration')
        plt.grid(True)
        plt.legend()

        # Plot jerk
        plt.subplot(3, 1, 3)
        plt.plot(time_values_jerk, jerk_processed, label='Jerk', linestyle='-')
        plt.title(f'{name}-S{n}-S{m}-T{i}- Jerk')
        plt.xlabel('Time (s)')
        plt.ylabel('Jerk')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()
    else:
        print(f"Failed to plot data for {name}_S{n}_S{m}_T{i}")

# Example usage
#plot_velocity_acceleration_jerk("Star", 1, 1, 1)




def plot_all_trials(name, n, m):
    """
    Plot tangential velocities (v_xy) for all trials of a given subject and session.
    """
    plt.figure(figsize=(15, 10))
    for i in range(1, 6):  # Assuming 5 trials per session
        result = extract_velocity(name, n, m, i)
        if result is not None:
            x_positions, y_positions, v_x, v_y, v_xy, time_values = result
            v_xy_processed = process_velocity(v_xy,time_values)
            plt.subplot(3, 2, i)
            plt.plot(time_values[:-1], v_xy_processed[:-1], linestyle='-')  # Continuous line without points
            plt.title(f'Trial{i}')
            plt.xlabel('Time (s)')
            plt.ylabel('V (cm/s)')
            plt.grid(True)

    plt.tight_layout()
    plt.suptitle(f'Subject{n}-Velocity Profiles of Various Trials of First Session for {name}', y=1.02)
    plt.subplots_adjust(top=0.9)
    plt.show()

"""plot_all_trials("B", 12, 1)
plot_all_trials("B", 12, 7)"""

def calculate_mean_speed(name, m, i, n=range(1, 13)):
    #calculate the avarage of each shape, session, and trial across all subjects.
    speeds = []

    for subject in n:
        result = extract_velocity(name, subject, m, i)
        if result is not None:
            x_positions, y_positions, v_x, v_y, v_xy, time_values = result
            v_xy_processed = process_velocity(v_xy, time_values)
            mean_speed = np.nanmean(v_xy_processed)
            #print(f"Subject: {subject}, Session: {m}, Trial: {i}, Mean Speed: {mean_speed}")
            speeds.append(mean_speed)

    overall_mean = np.nanmean(speeds)
    overall_std = np.nanstd(speeds)
    #print(f"Session {m}: Overall Mean Speed = {overall_mean}, Std Dev = {overall_std}")
    return overall_mean, overall_std

def plot_mean_speed_across_sessions(shapes, sessions, trials, subjects=range(1, 13)):
    plt.figure(figsize=(10, 6))

    for shape in shapes:
        mean_speeds = []
        std_devs = []

        for session in sessions:
            session_mean_speeds = []
            session_std_devs = []
            for trial in trials:
                mean_speed, std_dev = calculate_mean_speed(shape, session, trial, subjects)
                session_mean_speeds.append(mean_speed)
                session_std_devs.append(std_dev)

            # Calculate the mean and std deviation across trials for the session
            mean_speeds.append(np.nanmean(session_mean_speeds))
            std_devs.append(np.nanmean(session_std_devs))

        # Plot with error bars
        plt.errorbar(sessions, mean_speeds/np.max(mean_speeds), yerr=std_devs, label=shape, marker='o', linestyle='--')

    plt.title('Mean Speed with Error Bars Across Sessions for Different Shapes')
    plt.xlabel('Sessions')
    plt.ylabel('Mean Speed')
    plt.legend(title="Shapes")
    plt.grid(True)
    plt.show()

def calculate_mean_acceleration(name, m, i, n=range(1, 13)):
    accelerations = []

    for subject in n:
        result = extract_velocity(name, subject, m, i)
        if result is not None:
            x_positions, y_positions, v_x, v_y, v_xy, time_values = result
            v_xy_processed = process_velocity(v_xy, time_values)
            acceleration = extract_acceleration(v_xy_processed, time_values)
            acceleration_processed = process_acceleration(acceleration, time_values)
            mean_acceleration = np.nanmean(acceleration_processed)
            accelerations.append(mean_acceleration)

    overall_mean_acceleration = np.nanmean(accelerations)
    overall_std_acceleration = np.nanstd(accelerations)
    return overall_mean_acceleration, overall_std_acceleration


def plot_mean_acceleration_across_sessions(shapes, sessions, trials, subjects=range(1, 13)):
    plt.figure(figsize=(10, 6))

    for shape in shapes:
        mean_accelerations = []
        std_devs = []

        for session in sessions:
            session_mean_accelerations = []
            session_std_devs = []
            for trial in trials:
                mean_acceleration, std_dev = calculate_mean_acceleration(shape, session, trial, subjects)
                session_mean_accelerations.append(mean_acceleration)
                session_std_devs.append(std_dev)

            # Calculate the mean and std deviation across trials for the session
            mean_accelerations.append(np.nanmean(session_mean_accelerations))
            std_devs.append(np.nanmean(session_std_devs))

        # Plot with error bars
        plt.errorbar(sessions, mean_accelerations, yerr=std_devs, label=shape, marker='o', linestyle='--')

    plt.title('Mean Acceleration with Error Bars Across Sessions for Different Shapes')
    plt.xlabel('Sessions')
    plt.ylabel('Mean Acceleration')
    plt.legend(title="Shapes")
    plt.grid(True)
    plt.show()



def calculate_mean_jerk(name, m, i, n=range(1, 13)):
    jerks = []

    for subject in n:
        result = extract_velocity(name, subject, m, i)
        if result is not None:
            x_positions, y_positions, v_x, v_y, v_xy, time_values = result
            v_xy_processed = process_velocity(v_xy, time_values)
            acceleration = extract_acceleration(v_xy_processed, time_values)
            acceleration_processed = process_acceleration(acceleration, time_values)
            jerk = extract_jerk(acceleration_processed, time_values)
            jerk_processed = process_jerk(jerk, time_values)
            mean_jerk = np.nanmean(jerk_processed)
            jerks.append(mean_jerk)

    overall_mean_jerk = np.nanmean(jerks)
    overall_std_jerk = np.nanstd(jerks)
    return overall_mean_jerk, overall_std_jerk


def plot_mean_jerk_across_sessions(shapes, sessions, trials, subjects=range(1, 13)):
    plt.figure(figsize=(10, 6))

    for shape in shapes:
        mean_jerks = []
        std_devs = []

        for session in sessions:
            session_mean_jerks = []
            session_std_devs = []
            for trial in trials:
                mean_jerk, std_dev = calculate_mean_jerk(shape, session, trial, subjects)
                session_mean_jerks.append(mean_jerk)
                session_std_devs.append(std_dev)

            # Calculate the mean and std deviation across trials for the session
            mean_jerks.append(np.nanmean(session_mean_jerks))
            std_devs.append(np.nanmean(session_std_devs))

        # Plot with error bars
        plt.errorbar(sessions, mean_jerks, yerr=std_devs, label=shape, marker='o', linestyle='--')

    plt.title('Mean Jerk with Error Bars Across Sessions for Different Shapes')
    plt.xlabel('Sessions')
    plt.ylabel('Mean Jerk')
    plt.legend(title="Shapes")
    plt.grid(True)
    plt.show()



# Example usage
shapes = ["B", "M", "O", "Star", "Clover"]
#shapes = ["Star"]
sessions = range (1,8)
trials = range(1, 6)
#plot_mean_speed_across_sessions(shapes, sessions, trials)
#plot_mean_acceleration_across_sessions(shapes, sessions, trials)
#plot_mean_jerk_across_sessions(shapes, sessions, trials)"""

def calculate_normalized_mean_speed(name, m, i, n=range(1, 13)):
    normalized_speeds = []

    for subject in n:
        result = extract_velocity(name, subject, m, i)
        if result is not None:
            x_positions, y_positions, v_x, v_y, v_xy, time_values = result
            #v_xy_processed = process_velocity(v_xy, time_values)
            v_xy_processed = v_xy
            mean_speed = np.nanmean(v_xy_processed)
            max_speed = np.nanmax(v_xy_processed)

            if max_speed > 0:  # Avoid division by zero
                normalized_mean_speed = mean_speed / max_speed
                normalized_speeds.append(normalized_mean_speed)

    overall_normalized_mean = np.nanmean(normalized_speeds)
    overall_std = np.nanstd(normalized_speeds)
    return overall_normalized_mean, overall_std


def plot_normalized_mean_speed_across_sessions(shapes, sessions, trials, subjects=range(1, 13)):
    plt.figure(figsize=(10, 6))

    for shape in shapes:
        normalized_mean_speeds = []
        std_devs = []

        for session in sessions:
            session_normalized_mean_speeds = []
            session_std_devs = []
            for trial in trials:
                normalized_mean_speed, std_dev = calculate_normalized_mean_speed(shape, session, trial, subjects)
                session_normalized_mean_speeds.append(normalized_mean_speed)
                session_std_devs.append(std_dev)

            # Calculate the mean and std deviation across trials for the session
            normalized_mean_speeds.append(np.nanmean(session_normalized_mean_speeds))
            std_devs.append(np.nanmean(session_std_devs))

        # Plot with error bars
        plt.errorbar(sessions, normalized_mean_speeds, yerr=std_devs, label=shape, marker='o', linestyle='--')

    plt.title('Normalized Mean Speed with Error Bars Across Sessions for Different Shapes')
    plt.xlabel('Sessions')
    plt.ylabel('Normalized Mean Speed')
    plt.legend(title="Shapes")
    plt.grid(True)
    plt.show()




# %%
#for comparison version

def calculate_and_plot_normalized_mean_speed(shapes=['B','M','O','Star','Clover'],subjects=range(1, 13), sessions= range(1,8), trials=range(1,6), show = True):
    #plt.figure(figsize=(10, 6))
    #print(subjects)
    if show:
      fig, axs = plt.subplots(1, 1, figsize=(8, 5))
    for shape in shapes:
        shape_mean_speeds = []
        shape_mean2_speeds = []
        shape_mean_speeds_normalize = []
        shape_mean2_speeds_normalize = []

        std_devs = []
        std_max_devs = []
        for subject in subjects:
            subject_mean_speeds = []
            subject_mean2_speeds = []
            subject_mean_speeds_normalize = []
            subject_mean2_speeds_normalize = []

            for trial in trials:
                trial_mean_speeds = []
                trial_mean2_speeds = []
                trial_mean_speeds_normalize = []
                trial_mean2_speeds_normalize = []

                for session in sessions:
                    result = extract_velocity(shape, subject, session, trial)
                    result2 = extract_velocity_previous(shape, subject , session, trial)
                    if result is not None:
                        x_positions, y_positions, v_x, v_y, v_xy, time_values = result
                        x_positions2, y_positions2, v_x2, v_y2, v_xy2, time_values2 = result2
                        #v_xy_processed = process_velocity(v_xy, time_values)
                        mean_speed = np.nanmean(v_xy)
                        mean2_speed = np.nanmean(v_xy2)


                        trial_mean_speeds.append(mean_speed)
                        trial_mean2_speeds.append(mean2_speed)


                    else:
                        trial_mean_speeds.append(np.nan)
                        trial_mean2_speeds.append(np.nan)



                trial_mean_speeds_normalize = (trial_mean_speeds - np.nanmin(trial_mean_speeds))/(np.nanmax(trial_mean_speeds) - np.nanmin(trial_mean_speeds))
                trials_mean2_speeds_normalize = (trial_mean2_speeds - np.nanmin(trial_mean2_speeds))/(np.nanmax(trial_mean2_speeds) - np.nanmin(trial_mean2_speeds))


                subject_mean_speeds.append(trial_mean_speeds)
                subject_mean2_speeds.append(trial_mean2_speeds)
                subject_mean_speeds_normalize.append(trial_mean_speeds_normalize)
                subject_mean2_speeds_normalize.append(trials_mean2_speeds_normalize)


                mean_subject_mean_speeds = np.nanmean(subject_mean_speeds, axis =0)
                mean_subject_mean2_speeds = np.nanmean(subject_mean2_speeds, axis = 0)
                mean_subject_mean_speeds_normalize = np.nanmean(subject_mean_speeds_normalize, axis =0)
                mean_subject_mean2_speeds_normalize = np.nanmean(subject_mean2_speeds_normalize, axis = 0)


                #std_subject_mean_speeds = np.nanstd(subject_mean_speeds, axis = 0)
                #std_subject_maxs_speeds = np.nanstd(subject_max_speeds, axis = 0)
            shape_mean_speeds.append(mean_subject_mean_speeds)
            shape_mean2_speeds.append(mean_subject_mean2_speeds)
            shape_mean_speeds_normalize.append(mean_subject_mean_speeds_normalize)
            shape_mean2_speeds_normalize.append(mean_subject_mean2_speeds_normalize)


            #std_devs.append(std_subject_mean_speeds)
            #std_max_devs.append(std_subject_maxs_speeds)

        means = (np.nanmean(shape_mean_speeds, axis = 0))
        mean2s = (np.nanmean(shape_mean2_speeds, axis = 0))
        means_normalize = (np.nanmean(shape_mean_speeds_normalize, axis = 0))
        mean2s_normalize = (np.nanmean(shape_mean2_speeds_normalize, axis = 0))

        #std_devs =
        if show == 1 :
          """ axs[0][0].plot(sessions, means, label=shape, marker='o', linestyle='--')
          axs[0][0].set_title('(higher order) Across Sessions for Different Shapes')
          axs[0][0].set_xlabel('Sessions')
          axs[0][0].set_ylabel('Mean Speed')
          axs[0][0].legend(title="Shapes")
          axs[0][0].grid(True)

          axs[0][1].plot(sessions, mean2s, label=shape, marker='o', linestyle='--')
          axs[0][1].set_title(' (first order) Across Sessions for Different Shapes')
          axs[0][1].set_xlabel('Sessions')
          axs[0][1].set_ylabel('Mean Speed')
          axs[0][1].legend(title="Shapes")
          axs[0][1].grid(True)"""

          axs.plot(sessions, means_normalize, label=shape, marker='o', linestyle='--')
          axs.set_title('Normalized Mean Velocity (5-Point Stencil Method) Across Sessions for Different Shapes')
          axs.set_xlabel('Sessions')
          axs.set_ylabel('Normalized Mean Velocity')
          axs.legend(title="Shapes")
          axs.grid(True)


          """axs.plot(sessions, mean2s_normalize, label=shape, marker='o', linestyle='--')
          axs.set_title('Normalized Mean Velocity Across Sessions for Different Shapes')
          axs.set_xlabel('Sessions')
          axs.set_ylabel('Normalized Mean Velocity')
          axs.legend(title="Shapes")
          axs.grid(True)"""

          """axs[2].plot(sessions, stds, label=shape, marker='o', linestyle='--')
          axs[2].set_title('Std Speed Across Sessions for Different Shapes')
          axs[2].set_xlabel('Sessions')
          axs[2].set_ylabel('Normalized Std Speed')
          axs[2].legend(title="Shapes")
          axs[2].grid(True)"""

    if show == True:
      plt.tight_layout()
      plt.show()
    return means , mean2s

# Example usage
"""shapes = ["B", "M", "O", "Star", "Clover"]
#shapes = ["B"]
sessions = range(1, 8)
#sessions = range(1, 2)
trials = range(1, 6)
#trials = range(5,6)
#subjects = range(1,3)"""
calculate_and_plot_normalized_mean_speed()


# %%
#mean squared jerk
def calculate_and_plot_normalized_mean_speed(shapes=['B','M','O','Star','Clover'],subjects=range(1, 13), sessions= range(1,8), trials=range(1,6), show = True):
    #plt.figure(figsize=(10, 6))
    #print(subjects)
    if show:
      fig, axs = plt.subplots(4, 1, figsize=(10, 24))
    for shape in shapes:
        shape_max_speeds = []
        shape_mean_speeds = []
        shape_std_speeds = []
        shape_mean_squared_jerk = []

        std_devs = []
        std_max_devs = []
        for subject in subjects:
            subject_mean_speeds = []
            subject_max_speeds = []
            subject_std_speeds = []
            subject_mean_squared_jerk = []

            for trial in trials:
                trial_mean_speeds = []
                trial_max_speeds = []
                trial_std_speeds = []
                trial_mean_squared_jerk = []

                for session in sessions:
                    result = extract_velocity(shape, subject, session, trial)
                    if result is not None:
                        x_positions, y_positions, v_x, v_y, v_xy, time_values = result
                        #v_xy_processed = process_velocity(v_xy, time_values)
                        mean_speed = np.nanmean(v_xy)
                        max_speed = np.nanmax(v_xy)
                        acceleration = np.diff(v_xy) / np.diff(time_values[0:len(v_xy)])
                        jerk = np.diff(acceleration) / np.diff(time_values[0:len(acceleration)])
                        squared_jerk = jerk ** 2
                        t1= time_values[1]
                        t2 = time_values[-1]
                        mean_squared_jerk = (np.trapz(squared_jerk, time_values[0:len(squared_jerk)]) / (t2-t1))

                        std_speed = np.nanstd(v_xy)

                        trial_mean_speeds.append(mean_speed)
                        trial_max_speeds.append(max_speed)
                        trial_std_speeds.append(std_speed)
                        trial_mean_squared_jerk.append(mean_squared_jerk)

                    else:
                        trial_mean_speeds.append(np.nan)
                        trial_max_speeds.append(np.nan)
                        trial_std_speeds.append(np.nan)
                        trial_mean_squared_jerk.append(np.nan)


                trial_mean_speeds = (trial_mean_speeds - np.nanmin(trial_mean_speeds))/(np.nanmax(trial_mean_speeds) - np.nanmin(trial_mean_speeds))
                subject_mean_speeds.append(trial_mean_speeds)
                subject_max_speeds.append(trial_max_speeds)
                subject_std_speeds.append(trial_std_speeds)
                subject_mean_squared_jerk.append(trial_mean_squared_jerk)

                mean_subject_mean_speeds = np.nanmean(subject_mean_speeds, axis =0)
                mean_subject_max_speeds = np.nanmean(subject_max_speeds, axis = 0)
                mean_subject_std_speeds = np.nanmean(subject_std_speeds, axis = 0)
                mean_subject_mean_squared_jerk = np.nanmean(subject_mean_squared_jerk, axis = 0)

                #std_subject_mean_speeds = np.nanstd(subject_mean_speeds, axis = 0)
                #std_subject_maxs_speeds = np.nanstd(subject_max_speeds, axis = 0)
            shape_mean_speeds.append(mean_subject_mean_speeds)
            shape_max_speeds.append(mean_subject_max_speeds)
            shape_std_speeds.append(mean_subject_std_speeds)
            shape_mean_squared_jerk.append(mean_subject_mean_squared_jerk)

            #std_devs.append(std_subject_mean_speeds)
            #std_max_devs.append(std_subject_maxs_speeds)

        means = (np.nanmean(shape_mean_speeds, axis = 0))
        maxs = (np.nanmean(shape_max_speeds, axis = 0))
        stds = (np.nanmean(shape_std_speeds, axis = 0))
        mean_squared_jerks = (np.nanmean(shape_mean_squared_jerk, axis = 0))
        #std_devs =
        if show == 1 :
          axs[0].plot(sessions, means, label=shape, marker='o', linestyle='--')
          axs[0].set_title('Normalized Mean Speed Across Sessions for Different Shapes')
          axs[0].set_xlabel('Sessions')
          axs[0].set_ylabel('Normalized Mean Speed')
          axs[0].legend(title="Shapes")
          axs[0].grid(True)

          axs[1].plot(sessions, maxs, label=shape, marker='o', linestyle='--')
          axs[1].set_title('Max Speed Across Sessions for Different Shapes')
          axs[1].set_xlabel('Sessions')
          axs[1].set_ylabel('Normalized Max Speed')
          axs[1].legend(title="Shapes")
          axs[1].grid(True)

          axs[2].plot(sessions, stds, label=shape, marker='o', linestyle='--')
          axs[2].set_title('Std Speed Across Sessions for Different Shapes')
          axs[2].set_xlabel('Sessions')
          axs[2].set_ylabel('Normalized Std Speed')
          axs[2].legend(title="Shapes")
          axs[2].grid(True)


          axs[3].plot(sessions, mean_squared_jerks, label=shape, marker='o', linestyle='--')
          axs[3].set_title('Mean Squared Jerk Across Sessions for Different Shapes')
          axs[3].set_xlabel('Sessions')
          axs[3].set_ylabel('Mean Squared Jerk')
          axs[3].legend(title="Shapes")
          axs[3].grid(True)

    if show == True:
      plt.tight_layout()
      plt.show()
    return means , maxs , stds

# Example usage
"""shapes = ["B", "M", "O", "Star", "Clover"]
#shapes = ["B"]
sessions = range(1, 8)
#sessions = range(1, 2)
trials = range(1, 6)
#trials = range(5,6)
#subjects = range(1,3)"""
calculate_and_plot_normalized_mean_speed()


# %%
#accumulative squared jerk


def calculate_and_plot_normalized_mean_speed(shapes=['B','M','O','Star','Clover'],subjects=range(1, 13), sessions= range(1,8), trials=range(1,6), show = True):
    #plt.figure(figsize=(10, 6))
    #print(subjects)
    if show:
      fig, axs = plt.subplots(4, 1, figsize=(10, 24))
    for shape in shapes:
        shape_max_speeds = []
        shape_mean_speeds = []
        shape_std_speeds = []
        shape_mean_squared_jerk = []

        std_devs = []
        std_max_devs = []
        for subject in subjects:
            subject_mean_speeds = []
            subject_max_speeds = []
            subject_std_speeds = []
            subject_mean_squared_jerk = []

            for trial in trials:
                trial_mean_speeds = []
                trial_max_speeds = []
                trial_std_speeds = []
                trial_mean_squared_jerk = []

                for session in sessions:
                    result = extract_velocity(shape, subject, session, trial)
                    if result is not None:
                        x_positions, y_positions, v_x, v_y, v_xy, time_values = result
                        #v_xy_processed = process_velocity(v_xy, time_values)
                        mean_speed = np.nanmean(v_xy)
                        max_speed = np.nanmax(v_xy)
                        acceleration = np.diff(v_xy) / np.diff(time_values[0:len(v_xy)])
                        jerk = np.diff(acceleration) / np.diff(time_values[0:len(acceleration)])
                        squared_jerk = jerk ** 2
                        mean_squared_jerk = np.sum(squared_jerk)

                        std_speed = np.nanstd(v_xy)

                        trial_mean_speeds.append(mean_speed)
                        trial_max_speeds.append(max_speed)
                        trial_std_speeds.append(std_speed)
                        trial_mean_squared_jerk.append(mean_squared_jerk)

                    else:
                        trial_mean_speeds.append(np.nan)
                        trial_max_speeds.append(np.nan)
                        trial_std_speeds.append(np.nan)
                        trial_mean_squared_jerk.append(np.nan)


                trial_mean_speeds = (trial_mean_speeds - np.nanmin(trial_mean_speeds))/(np.nanmax(trial_mean_speeds) - np.nanmin(trial_mean_speeds))
                subject_mean_speeds.append(trial_mean_speeds)
                subject_max_speeds.append(trial_max_speeds)
                subject_std_speeds.append(trial_std_speeds)
                subject_mean_squared_jerk.append(trial_mean_squared_jerk)

                mean_subject_mean_speeds = np.nanmean(subject_mean_speeds, axis =0)
                mean_subject_max_speeds = np.nanmean(subject_max_speeds, axis = 0)
                mean_subject_std_speeds = np.nanmean(subject_std_speeds, axis = 0)
                mean_subject_mean_squared_jerk = np.nanmean(subject_mean_squared_jerk, axis = 0)

                #std_subject_mean_speeds = np.nanstd(subject_mean_speeds, axis = 0)
                #std_subject_maxs_speeds = np.nanstd(subject_max_speeds, axis = 0)
            shape_mean_speeds.append(mean_subject_mean_speeds)
            shape_max_speeds.append(mean_subject_max_speeds)
            shape_std_speeds.append(mean_subject_std_speeds)
            shape_mean_squared_jerk.append(mean_subject_mean_squared_jerk)

            #std_devs.append(std_subject_mean_speeds)
            #std_max_devs.append(std_subject_maxs_speeds)

        means = (np.nanmean(shape_mean_speeds, axis = 0))
        maxs = (np.nanmean(shape_max_speeds, axis = 0))
        stds = (np.nanmean(shape_std_speeds, axis = 0))
        mean_squared_jerks = (np.nanmean(shape_mean_squared_jerk, axis = 0))
        #std_devs =
        if show == 1 :
          axs[0].plot(sessions, means, label=shape, marker='o', linestyle='--')
          axs[0].set_title('Normalized Mean Speed Across Sessions for Different Shapes')
          axs[0].set_xlabel('Sessions')
          axs[0].set_ylabel('Normalized Mean Speed')
          axs[0].legend(title="Shapes")
          axs[0].grid(True)

          axs[1].plot(sessions, maxs, label=shape, marker='o', linestyle='--')
          axs[1].set_title('Max Speed Across Sessions for Different Shapes')
          axs[1].set_xlabel('Sessions')
          axs[1].set_ylabel('Normalized Max Speed')
          axs[1].legend(title="Shapes")
          axs[1].grid(True)

          axs[2].plot(sessions, stds, label=shape, marker='o', linestyle='--')
          axs[2].set_title('Std Speed Across Sessions for Different Shapes')
          axs[2].set_xlabel('Sessions')
          axs[2].set_ylabel('Normalized Std Speed')
          axs[2].legend(title="Shapes")
          axs[2].grid(True)


          axs[3].plot(sessions, mean_squared_jerks, label=shape, marker='o', linestyle='--')
          axs[3].set_title('Mean Squared Jerk Across Sessions for Different Shapes')
          axs[3].set_xlabel('Sessions')
          axs[3].set_ylabel('Mean Squared Jerk')
          axs[3].legend(title="Shapes")
          axs[3].grid(True)

    if show == True:
      plt.tight_layout()
      plt.show()
    return means , maxs , stds

# Example usage
"""shapes = ["B", "M", "O", "Star", "Clover"]
#shapes = ["B"]
sessions = range(1, 8)
#sessions = range(1, 2)
trials = range(1, 6)
#trials = range(5,6)
#subjects = range(1,3)"""
calculate_and_plot_normalized_mean_speed()


# %%
#mean squared jerk


def calculate_and_plot_normalized_mean_speed(shapes=['B','M','O','Star','Clover'],subjects=range(1, 13), sessions= range(1,8), trials=range(1,6), show = True):
    #plt.figure(figsize=(10, 6))
    #print(subjects)
    if show:
      fig, axs = plt.subplots(4, 1, figsize=(10, 24))
    for shape in shapes:
        shape_max_speeds = []
        shape_mean_speeds = []
        shape_std_speeds = []
        shape_mean_squared_jerk = []

        std_devs = []
        std_max_devs = []
        for subject in subjects:
            subject_mean_speeds = []
            subject_max_speeds = []
            subject_std_speeds = []
            subject_mean_squared_jerk = []

            for trial in trials:
                trial_mean_speeds = []
                trial_max_speeds = []
                trial_std_speeds = []
                trial_mean_squared_jerk = []

                for session in sessions:
                    result = extract_velocity(shape, subject, session, trial)
                    if result is not None:
                        x_positions, y_positions, v_x, v_y, v_xy, time_values = result
                        #v_xy_processed = process_velocity(v_xy, time_values)
                        mean_speed = np.nanmean(v_xy)
                        max_speed = np.nanmax(v_xy)
                        std_speed = np.nanstd(v_xy)

                        acceleration = np.zeros(len(v_xy))
                        dt = np.diff(time_values)
                        dt = np.concatenate(([dt[0]], dt, [dt[-1]]))  # Extend dt for consistent length

                        for i in range(2, len(v_xy) - 2):
                            acceleration[i] = (-v_xy[i - 2] + 8 * v_xy[i - 1] - 8 * v_xy[i + 1] + v_xy[i + 2]) / (12 * dt[i])

                        # Calculate jerk using 5-point stencil method
                        jerk = np.zeros(len(acceleration))
                        for i in range(2, len(acceleration) - 2):
                            jerk[i] = (-acceleration[i - 2] + 8 * acceleration[i - 1] - 8 * acceleration[i + 1] + acceleration[i + 2]) / (12 * dt[i])


                        squared_jerk = jerk ** 2
                        t1 = time_values[1]
                        t2 = time_values[-1]
                        mean_squared_jerk = (np.trapz(squared_jerk, time_values[0:len(squared_jerk)]) / (t2 - t1))

                        trial_mean_speeds.append(mean_speed)
                        trial_max_speeds.append(max_speed)
                        trial_std_speeds.append(std_speed)
                        trial_mean_squared_jerk.append(mean_squared_jerk)

                    else:
                        trial_mean_speeds.append(np.nan)
                        trial_max_speeds.append(np.nan)
                        trial_std_speeds.append(np.nan)
                        trial_mean_squared_jerk.append(np.nan)


                trial_mean_speeds = (trial_mean_speeds - np.nanmin(trial_mean_speeds))/(np.nanmax(trial_mean_speeds) - np.nanmin(trial_mean_speeds))
                subject_mean_speeds.append(trial_mean_speeds)
                subject_max_speeds.append(trial_max_speeds)
                subject_std_speeds.append(trial_std_speeds)
                subject_mean_squared_jerk.append(trial_mean_squared_jerk)

                mean_subject_mean_speeds = np.nanmean(subject_mean_speeds, axis =0)
                mean_subject_max_speeds = np.nanmean(subject_max_speeds, axis = 0)
                mean_subject_std_speeds = np.nanmean(subject_std_speeds, axis = 0)
                mean_subject_mean_squared_jerk = np.nanmean(subject_mean_squared_jerk, axis = 0)

                #std_subject_mean_speeds = np.nanstd(subject_mean_speeds, axis = 0)
                #std_subject_maxs_speeds = np.nanstd(subject_max_speeds, axis = 0)
            shape_mean_speeds.append(mean_subject_mean_speeds)
            shape_max_speeds.append(mean_subject_max_speeds)
            shape_std_speeds.append(mean_subject_std_speeds)
            shape_mean_squared_jerk.append(mean_subject_mean_squared_jerk)

            #std_devs.append(std_subject_mean_speeds)
            #std_max_devs.append(std_subject_maxs_speeds)

        means = (np.nanmean(shape_mean_speeds, axis = 0))
        maxs = (np.nanmean(shape_max_speeds, axis = 0))
        stds = (np.nanmean(shape_std_speeds, axis = 0))
        mean_squared_jerks = (np.nanmean(shape_mean_squared_jerk, axis = 0))
        #std_devs =
        if show == 1 :
          axs[0].plot(sessions, means, label=shape, marker='o', linestyle='--')
          axs[0].set_title('Normalized Mean Speed Across Sessions for Different Shapes')
          axs[0].set_xlabel('Sessions')
          axs[0].set_ylabel('Normalized Mean Speed')
          axs[0].legend(title="Shapes")
          axs[0].grid(True)

          axs[1].plot(sessions, maxs, label=shape, marker='o', linestyle='--')
          axs[1].set_title('Max Speed Across Sessions for Different Shapes')
          axs[1].set_xlabel('Sessions')
          axs[1].set_ylabel('Normalized Max Speed')
          axs[1].legend(title="Shapes")
          axs[1].grid(True)

          axs[2].plot(sessions, stds, label=shape, marker='o', linestyle='--')
          axs[2].set_title('Std Speed Across Sessions for Different Shapes')
          axs[2].set_xlabel('Sessions')
          axs[2].set_ylabel('Normalized Std Speed')
          axs[2].legend(title="Shapes")
          axs[2].grid(True)


          axs[3].plot(sessions, mean_squared_jerks, label=shape, marker='o', linestyle='--')
          axs[3].set_title('Mean Squared Jerk Across Sessions for Different Shapes')
          axs[3].set_xlabel('Sessions')
          axs[3].set_ylabel('Mean Squared Jerk')
          axs[3].legend(title="Shapes")
          axs[3].grid(True)

    if show == True:
      plt.tight_layout()
      plt.show()
    return means , maxs , stds

# Example usage
"""shapes = ["B", "M", "O", "Star", "Clover"]
#shapes = ["B"]
sessions = range(1, 8)
#sessions = range(1, 2)
trials = range(1, 6)
#trials = range(5,6)
#subjects = range(1,3)"""
calculate_and_plot_normalized_mean_speed()


# %%
#real version!!!!!!!!


def calculate_and_plot_normalized_mean_speed(shapes=['B','M','O','Star','Clover'],subjects=range(1, 13), sessions= range(1,8), trials=range(1,6), show = True):
    #plt.figure(figsize=(10, 6))
    #print(subjects)
    if show:
      fig, axs = plt.subplots(3, 1, figsize=(10, 18))
    for shape in shapes:
        shape_max_speeds = []
        shape_mean_speeds = []
        shape_std_speeds = []

        std_devs = []
        std_max_devs = []
        for subject in subjects:
            subject_mean_speeds = []
            subject_max_speeds = []
            subject_std_speeds = []

            for trial in trials:
                trial_mean_speeds = []
                trial_max_speeds = []
                trial_std_speeds = []

                for session in sessions:
                    result = extract_velocity(shape, subject, session, trial)
                    if result is not None:
                        x_positions, y_positions, v_x, v_y, v_xy, time_values = result
                        #v_xy_processed = process_velocity(v_xy, time_values)
                        mean_speed = np.nanmean(v_xy)
                        max_speed = np.nanmax(v_xy)
                        std_speed = np.nanstd(v_xy)

                        trial_mean_speeds.append(mean_speed)
                        trial_max_speeds.append(max_speed)
                        trial_std_speeds.append(std_speed)

                    else:
                        trial_mean_speeds.append(np.nan)
                        trial_max_speeds.append(np.nan)
                        trial_std_speeds.append(np.nan)


                #trial_mean_speeds = (trial_mean_speeds - np.nanmin(trial_mean_speeds))/(np.nanmax(trial_mean_speeds) - np.nanmin(trial_mean_speeds))
                subject_mean_speeds.append(trial_mean_speeds)
                subject_max_speeds.append(trial_max_speeds)
                subject_std_speeds.append(trial_std_speeds)

                mean_subject_mean_speeds = np.nanmean(subject_mean_speeds, axis =0)
                mean_subject_max_speeds = np.nanmean(subject_max_speeds, axis = 0)
                mean_subject_std_speeds = np.nanmean(subject_std_speeds, axis = 0)

                #std_subject_mean_speeds = np.nanstd(subject_mean_speeds, axis = 0)
                #std_subject_maxs_speeds = np.nanstd(subject_max_speeds, axis = 0)
            shape_mean_speeds.append(mean_subject_mean_speeds)
            shape_max_speeds.append(mean_subject_max_speeds)
            shape_std_speeds.append(mean_subject_std_speeds)

            #std_devs.append(std_subject_mean_speeds)
            #std_max_devs.append(std_subject_maxs_speeds)

        means = (np.nanmean(shape_mean_speeds, axis = 0))
        maxs = (np.nanmean(shape_max_speeds, axis = 0))
        stds = (np.nanmean(shape_std_speeds, axis = 0))
        #std_devs =
        if show == 1 :
          axs[0].plot(sessions, means, label=shape, marker='o', linestyle='--')
          axs[0].set_title('Normalized Mean Speed Across Sessions for Different Shapes')
          axs[0].set_xlabel('Sessions')
          axs[0].set_ylabel('Normalized Mean Speed')
          axs[0].legend(title="Shapes")
          axs[0].grid(True)

          axs[1].plot(sessions, maxs, label=shape, marker='o', linestyle='--')
          axs[1].set_title('Max Speed Across Sessions for Different Shapes')
          axs[1].set_xlabel('Sessions')
          axs[1].set_ylabel('Normalized Max Speed')
          axs[1].legend(title="Shapes")
          axs[1].grid(True)

          axs[2].plot(sessions, stds, label=shape, marker='o', linestyle='--')
          axs[2].set_title('Std Speed Across Sessions for Different Shapes')
          axs[2].set_xlabel('Sessions')
          axs[2].set_ylabel('Normalized Std Speed')
          axs[2].legend(title="Shapes")
          axs[2].grid(True)

    if show == True:
      plt.tight_layout()
      plt.show()
    return means , maxs , stds

# Example usage
"""shapes = ["B", "M", "O", "Star", "Clover"]
#shapes = ["B"]
sessions = range(1, 8)
#sessions = range(1, 2)
trials = range(1, 6)
#trials = range(5,6)
#subjects = range(1,3)"""
calculate_and_plot_normalized_mean_speed()


# %%
"""from skimage.metrics import structural_similarity as ssim
import cv2
import os

def compare_images(img1_path, img2_path):
    # Load the two images in grayscale
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # Resize to ensure both images are of the same size
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    # Compute SSIM between the two images
    score, _ = ssim(img1, img2, full=True)
    return score

# Path to your ideal "B"
ideal_b_path = '/content/drive/MyDrive/project/pics/ideal_B.jpg'

# Path to the directory containing your handwritten "B"s
handwritten_b_dir = '/content/drive/MyDrive/project/pics'

# List to hold the similarity scores
similarities = []

# Loop through all handwritten "B"s and compare each one to the ideal "B"
for filename in os.listdir(handwritten_b_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # You can modify the extensions as neede
        handwritten_b_path = os.path.join(handwritten_b_dir, filename)
        score = compare_images(ideal_b_path, handwritten_b_path)
        similarities.append((filename, score))

# Sort the list by similarity score (from highest to lowest)
similarities.sort(key=lambda x: x[1], reverse=True)

# Display the results
for filename, score in similarities:
    print(f'{filename} has a similarity score of {score}')

# The most similar image will be the one with the highest SSIM score.
print(f"The most similar handwritten 'B' is: {similarities[0][0]} with a score of {similarities[0][1]}")"""

# %%
#Comparison SSIM
import cv2
import os
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt

def compare_images(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))  # Ensure the same size
    score, _ = ssim(img1, img2, full=True)
    return score

# Folder path where all the images are stored
images_dir = '/content/drive/MyDrive/project/pics'
shapes = [ 'B','M', 'O', 'Star', 'Clover']  # List your shapes here
#shapes =['Star']
sessions = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']

# Dictionary to hold average SSIM scores for each shape and session
average_scores = {shape: {session: [] for session in sessions} for shape in shapes}

# Ideal images for comparison (one per shape)
ideal_images = {
    'B': '/content/drive/MyDrive/project/pics1/ideal_B.jpg',
    'M': '/content/drive/MyDrive/project/pics1/ideal_M.jpg',
    'O': '/content/drive/MyDrive/project/pics1/ideal_O.jpg' ,
    'Star': '/content/drive/MyDrive/project/pics1/ideal_Star.jpg',
    'Clover': '/content/drive/MyDrive/project/pics1/ideal_Clover.jpg'
    # Add paths for other shapes
}

# Compute SSIM for each session and shape
i = 12
for shape in shapes:
    ideal_image_path = ideal_images[shape]  # Ideal image for this shape
    for session in sessions:
        subject_scores = []
        for subject in range(1,i+1):
                trial_scores = []
                for trial in range(1,6):
                  for filename in os.listdir(images_dir):
                    if filename.startswith(f"{shape}_{session}_S{subject}_T{trial}") and filename.endswith(".jpg"):
                        img_path = os.path.join(images_dir, filename)
                        score = compare_images(ideal_image_path, img_path)
                        #print(f"SSIM score for {filename}: {score}")
                        trial_scores.append(score)
                score = np.nanmean(trial_scores)
                #print(f"SSIM score for {filename}: {score}")
                subject_scores.append(score)
        if subject_scores:
            average_scores[shape][session] = np.nanmean(subject_scores)
            print(f"Average SSIM score for {shape} in session {session}: {average_scores[shape][session]}")
        else:
          average_scores[shape][session] = np.nan
# Plot the results
for shape in shapes:

    scores2 = [average_scores[shape][session] for session in sessions]
    mask = ~np.isnan(scores)
    print(mask)
    print(np.array(scores2)[mask])
    print(np.array(sessions)[mask])
    plt.plot(np.array(sessions)[mask], np.array(scores2)[mask],marker = 'o', label=f'{shape}')

plt.xlabel('Session')
plt.ylabel('Average SSIM Score')
plt.title('Average SSIM Score per Session for each Shape')
#plt.legend()
plt.show()


# %%
for shape in shapes:

    scores2 = [average_scores[shape][session] for session in sessions]
    mask = ~np.isnan(scores)
    #print(mask)
    scores = np.array(scores)[mask]
    scores2 = np.array(scores2)[mask]
    sessions = np.array(sessions)[mask]
    scores = (scores - scores.min()) / (scores.max() - scores.min())
    scores2 = (scores2 - scores2.min()) / (scores2.max() - scores2.min())
    print(scores)
    print(np.array(scores)[mask])
    print(np.array(sessions)[mask])
    plt.plot(sessions, scores2 ,marker = 'o', label=f'{shape}')

plt.xlabel('Session')
plt.ylabel('Normalized Average SSIM Score')
plt.title('Normalized Average SSIM Score per Session for each Shape')
plt.legend()
plt.show()

# %%
#Final  SSIM
import cv2
import os
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt

def compare_images(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))  # Ensure the same size
    score, _ = ssim(img1, img2, full=True)
    return score

# Folder path where all the images are stored
images_dir = '/content/drive/MyDrive/project/pics1'
shapes = [ 'B','M', 'O', 'Star','Clover']  # List your shapes here
#shapes =['Star']
sessions = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']

# Dictionary to hold average SSIM scores for each shape and session
average_scores = {shape: {session: [] for session in sessions} for shape in shapes}

# Ideal images for comparison (one per shape)
ideal_images = {
    'B': '/content/drive/MyDrive/project/pics1/ideal_B.jpg',
    'M': '/content/drive/MyDrive/project/pics1/ideal_M.jpg',
    'O': '/content/drive/MyDrive/project/pics1/ideal_O.jpg' ,
    'Star': '/content/drive/MyDrive/project/pics1/ideal_Star.jpg',
    'Clover': '/content/drive/MyDrive/project/pics1/ideal_Clover.jpg'
    # Add paths for other shapes
}

# Compute SSIM for each session and shape
i = 12
for shape in shapes:
    ideal_image_path = ideal_images[shape]  # Ideal image for this shape
    for session in sessions:
        session_scores = []
        for filename in os.listdir(images_dir):
          for subject in range(1,i+1):
            if filename.startswith(f"{shape}_{session}_S{subject}") and filename.endswith(".jpg"):
                img_path = os.path.join(images_dir, filename)
                score = compare_images(ideal_image_path, img_path)
                #print(f"SSIM score for {filename}: {score}")
                session_scores.append(score)
        if session_scores:
            average_scores[shape][session] = np.nanmean(session_scores)
            print(f"Average SSIM score for {shape} in session {session}: {average_scores[shape][session]}")
        else:
          average_scores[shape][session] = np.nan
# Plot the results
for shape in shapes:

    scores = [average_scores[shape][session] for session in sessions]
    mask = ~np.isnan(scores)
    print(mask)
    print(np.array(scores)[mask])
    print(np.array(sessions)[mask])
    plt.plot(np.array(sessions)[mask], np.array(scores)[mask],marker = 'o', label=f'{shape}')

plt.xlabel('Session')
plt.ylabel('Average SSIM Score')
plt.title('Average SSIM Score per Session for each Shape')
#plt.legend()
plt.show()


# %%
for shape in shapes:

    scores = [average_scores[shape][session] for session in sessions]
    mask = ~np.isnan(scores)
    #print(mask)
    scores = np.array(scores)[mask]
    sessions = np.array(sessions)[mask]
    scores = (scores - scores.min()) / (scores.max() - scores.min())
    print(scores)
    print(np.array(scores)[mask])
    print(np.array(sessions)[mask])
    plt.plot(sessions, scores ,marker = 'o', label=f'{shape}')

plt.xlabel('Session')
plt.ylabel('Normalized Average SSIM Score')
plt.title('Normalized Average SSIM Score per Session for each Shape')
plt.legend()
plt.show()

# %%
#mean time value

import pandas as pd
import matplotlib.pyplot as plt




plot_subject_session_files = range(6, 3)
plt.figure(figsize=(8, 5))
for shape in ['B', 'M', 'O', 'Star', 'Clover']:
    mean_times = []
    for session in range(1,8):
        session_mean_times = []
        for subject in range(1, 13):
            for trial in range(1,6):
                  row  = movement_time_df[(movement_time_df['Subject'] == subject) &(movement_time_df['Session'] == f'T{session}') & (movement_time_df['Shape'] == shape) &(movement_time_df['Trial'] == trial)]
                  if not row.empty:
                    time_value = row['Time'].values[0]
                    session_mean_times.append(time_value)
            # Calculate the mean and std deviation across trials for the session
        mean_times.append(np.nanmean(session_mean_times))

        # Plot with error bars

    plt.plot(sessions, mean_times, label=shape, marker='o', linestyle='--')

plt.title('Mean Time value Across Sessions for Different Shapes')
plt.xlabel('Sessions')
plt.ylabel('Mean Time Value')
plt.legend(title="Shapes")
plt.show()



# %%
import pandas as pd
import numpy as np
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# Create an empty list to store the data
data = []

excel_file_path = '/content/drive/MyDrive/project/Movement Time.xlsx'
smoothness_df_path = '/content/drive/MyDrive/project/data_frame.csv'
veloframe_path = '/content/drive/MyDrive/project/velocframe.csv'
finaldf_path = '/content/drive/MyDrive/project/findf.csv'


# Load the Excel file
movement_time_df = pd.read_excel(excel_file_path)
smoothness_df = pd.read_csv(smoothness_df_path)
velocity_df =  pd.read_csv(veloframe_path)
finaldf = pd.read_csv(finaldf_path)
finaldf['smoothness'] = (finaldf['smoothness'] - finaldf['smoothness'].min())/(finaldf['smoothness'].max() - finaldf['smoothness'].min())
finaldf['Curvature'] = (finaldf['Curvature'] - finaldf['Curvature'].min())/(finaldf['Curvature'].max() - finaldf['Curvature'].min())
finaldf['accelaration'] = (finaldf['accelaration'] - finaldf['accelaration'].min())/(finaldf['accelaration'].max() - finaldf['accelaration'].min())
finaldf['jerk'] = (finaldf['jerk'] - finaldf['jerk'].min())/(finaldf['jerk'].max() - finaldf['jerk'].min())



# Loop through shapes, subjects, sessions, and trials
for shape in shapes:
  for subject in range(1,13):
    for session in range (1, 8):
      if subject!= 6 or session !=5:
        # Construct the filename based on your naming convention
        filename = f"{shape}_S{session}_S{subject}.jpg"  # Adjust as needed
        file_path = os.path.join(images_dir, filename)

        # Check if the file exists
        if os.path.exists(file_path):
          # Get the SSIM score
          ideal_image_path = ideal_images[shape]
          ssim_score = compare_images(ideal_image_path, file_path)

          # Get the time value from your movement_time_df (assuming it's available)
          time_values = []
          smoothness =[]
          velocities = []
          row2 = smoothness_df[(smoothness_df['Subject'] == subject) & (smoothness_df['Session'] == session) & (smoothness_df['Shape'] == shape)]
          row3 = velocity_df[(velocity_df['Subject'] == subject) & (velocity_df['Session'] == session) & (velocity_df['Shape'] == shape)].dropna()
          row4 = finaldf[(finaldf['subject'] == subject) & (finaldf['session'] == session) & (finaldf['shape'] == shape)]
          #print(row3)

          for trial in range(1, 6):
            #print(shape , subject , session, trial)
            row = movement_time_df[(movement_time_df['Subject'] == subject) & (movement_time_df['Session'] == f'T{session}') & (movement_time_df['Shape'] == shape) & (movement_time_df['Trial'] == trial)]

            if not row.empty:
              time_values.append(row['Time'].values[0])
            else:
              time_values.append(np.nan)
          if time_values:
            mean_time_value = np.nanmean(time_values)
          else:
            mean_time_value = np,nan
          #print(shape, subject , session )
          smoothness_value = row4['smoothness'].values[0]
          velocity_value = row4['mean_V'].values[0]
          curvature = row4['Curvature'].values[0]
          acceleration = row4['accelaration'].values[0]
          jerk = row4['jerk'].values[0]
          #print(shape , subject , session, velocity_value)
          # Append the data to the list
          #means , maxs , stds

          mean_v , max_v , std_v = calculate_and_plot_normalized_mean_speed(shapes =[shape] , subjects = range(subject, subject+1) , sessions = range(session, session+1), show = False)
          # Append the data to the list
          data.append([shape, subject, session, ssim_score, smoothness_value, mean_time_value, velocity_value, curvature ,acceleration, jerk,mean_v[0], max_v[0] , std_v[0]])

# Create the DataFrame
df_init = pd.DataFrame(data, columns=['Shape', 'Subject', 'Session', 'SSIM_Score','Smoothness_Value','Time_Value' ,'Mean_Velocity','Curvature', 'Acceleration', 'Jerk', 'Mean_v', 'Max_v', 'Std_v'])

# Print the DataFrame
print(len(df_init))


# %%
#mean time value

import pandas as pd
import matplotlib.pyplot as plt




plot_subject_session_files = range(6, 3)
plt.figure(figsize=(8, 5))
for shape in ['B', 'M', 'O', 'Star', 'Clover']:
    mean_times = []
    for session in range(1,8):
        session_mean_times = []
        for subject in range(1, 13):
                  row  = finaldf[(finaldf['subject'] == subject) &(finaldf['session'] == session) & (finaldf['shape'] == shape)]
                  if not row.empty:
                    time_value = row['smoothness'].values[0]

                    session_mean_times.append(time_value)
            # Calculate the mean and std deviation across trials for the session
        mean_times.append(np.nanmean(session_mean_times))

        # Plot with error bars
    #mean_times = (mean_times - np.min(mean_times)) / (np.max(mean_times) - np.min(mean_times))
    plt.plot(sessions, mean_times, label=shape, marker='o', linestyle='--')

plt.title('Mean Smoothness value Across Sessions for Different Shapes')
plt.xlabel('Sessions')
plt.ylabel('Mean Smoothness Value')
plt.legend(title="Shapes")
plt.show()



# %%


df1_7 = df_init.copy()
df1_7 = df1_7[df1_7['Session'].isin([1, 7])]
##
df1_67 = df_init.copy()
df1_67.loc[df1_67['Session'] == 6, 'Session'] = 7
df1_67 = df1_67[df1_67['Session'].isin([1, 7])]
##
df1_567 = df_init.copy()
df1_567.loc[(df1_567['Session'] == 6) |(df1_567['Session'] == 5), 'Session'] = 7
df1_567 = df1_567[df1_567['Session'].isin([1, 7])]
###

df12_7 = df_init.copy()
df12_7.loc[df12_7['Session'] == 2, 'Session'] = 1
df12_7 = df12_7[df12_7['Session'].isin([1, 7])]

##
df12_67 = df_init.copy()
df12_67.loc[df12_67['Session'] == 2, 'Session'] = 1
df12_67.loc[df12_67['Session'] == 6, 'Session'] = 7
df12_67 = df12_67[df12_67['Session'].isin([1, 7])]
##
df12_567 = df_init.copy()
df12_567.loc[df12_567['Session'] == 2, 'Session'] = 1
df12_567.loc[(df12_567['Session'] == 5) |(df12_567['Session'] == 6), 'Session'] = 7
df12_567 = df12_567[df12_567['Session'].isin([1, 7])]
###


df123_7 = df_init.copy()
df123_7.loc[(df123_7['Session'] == 2) | (df123_7['Session'] == 3), 'Session'] = 1
df123_7 = df123_7[df123_7['Session'].isin([1, 7])]
##
df123_67 = df_init.copy()
df123_67.loc[(df123_67['Session'] == 2) | (df123_67['Session'] == 3), 'Session'] = 1
df123_67.loc[df123_67['Session'] == 6, 'Session'] = 7
df123_67 = df123_67[df123_67['Session'].isin([1, 7])]
##
df123_567 = df_init.copy()
df123_567.loc[(df123_567['Session'] == 2) | (df123_567['Session'] == 3), 'Session'] = 1
df123_567.loc[(df123_567['Session'] == 5) | (df123_567['Session'] == 6), 'Session'] = 7
df123_567 = df123_567[df123_567['Session'].isin([1, 7])]
###
df1234_567 = df_init.copy()
df1234_567.loc[(df1234_567['Session'] == 2) | (df1234_567['Session'] == 3) | (df1234_567['Session'] == 4), 'Session'] = 1
df1234_567.loc[(df1234_567['Session'] == 5) | (df1234_567['Session'] == 6), 'Session'] = 7
df1234_567 = df1234_567[df1234_567['Session'].isin([1, 7])]
##
df123_4567 = df_init.copy()
df123_4567.loc[(df123_4567['Session'] == 2) | (df123_4567['Session'] == 3), 'Session'] = 1
df123_4567.loc[(df123_4567['Session'] == 4) | (df123_4567['Session'] == 5) | (df123_4567['Session'] == 6), 'Session'] = 7
df123_4567 = df123_4567[df123_4567['Session'].isin([1, 7])]
##
######################
df1_3_7 = df_init.copy()
df1_3_7 = df1_3_7[df1_3_7['Session'].isin([1, 3, 7])]
##
df12_34_67 = df_init.copy()
df12_34_67.loc[df12_34_67['Session'] == 2, 'Session'] = 1
df12_34_67.loc[df12_34_67['Session'] == 4, 'Session'] = 3
df12_34_67.loc[df12_34_67['Session'] == 6, 'Session'] = 7
df12_34_67 = df12_34_67[df12_34_67['Session'].isin([1, 3, 7])]
##
df12_34_567 = df_init.copy()
df12_34_567.loc[df12_34_567['Session'] == 2, 'Session'] = 1
df12_34_567.loc[df12_34_567['Session'] == 4, 'Session'] = 3
df12_34_567.loc[(df12_34_567['Session'] == 5) | (df12_34_567['Session'] == 6), 'Session'] = 7
df12_34_567 = df12_34_567[df12_34_567['Session'].isin([1, 3, 7])]
##
df123_4_567 = df_init.copy()
df123_4_567.loc[(df123_4_567['Session'] == 2) | (df123_4_567['Session'] == 3), 'Session'] = 1
df123_4_567.loc[(df123_4_567['Session'] == 5) | (df123_4_567['Session'] == 6 ) , 'Session'] = 7
df123_4_567 = df123_4_567[df123_4_567['Session'].isin([1, 3, 7])]
##1-47
df123_45_67 = df_init.copy()
df123_45_67.loc[(df123_45_67['Session'] == 2) | (df123_45_67['Session'] == 3), 'Session'] = 1
df123_45_67.loc[(df123_45_67['Session'] == 5) , 'Session'] = 4
df123_45_67.loc[df123_45_67['Session'] == 6, 'Session'] = 7
df123_45_67 = df123_45_67[df123_45_67['Session'].isin([1, 4, 7])]
####
df12_345_67 = df_init.copy()
df12_345_67.loc[df12_345_67['Session'] == 2, 'Session'] = 1
df12_345_67.loc[(df12_345_67['Session'] == 4) | (df12_345_67['Session'] == 5), 'Session'] = 3
df12_345_67.loc[df12_345_67['Session'] == 6, 'Session'] = 7
df12_345_67 = df12_345_67[df12_345_67['Session'].isin([1, 3, 7])]
##
df1_6 = df_init.copy()
df1_6 = df1_6[df1_6['Session'].isin([1, 6])]
##
df1_2 = df_init.copy()
df1_2 = df1_2[df1_2['Session'].isin([1, 2])]
##
df1_56 = df_init.copy()
df1_56.loc[(df1_56['Session'] == 5), 'Session'] = 6
df1_56 = df1_56[df1_56['Session'].isin([1, 6])]

# %% [markdown]
# # Classification

# %%
###classification all shapes
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# List of features to use
features = [ 'Mean_Velocity','SSIM_Score', 'Smoothness_Value', 'Time_Value', 'Curvature', 'Acceleration', 'Jerk']
dataframes = [df1_2, df1_56, df1_6, df1_7, df1_67, df1_567, df12_7, df12_67, df12_567, df123_7, df123_67, df123_567, df1234_567,
              df123_4567, df1_3_7, df12_34_67, df12_34_567, df123_4_567, df12_345_67, df123_45_67]
dataframenames =['df1_2', 'df1_56', 'df1_6','df1_7', 'df1_67', 'df1_567', 'df12_7', 'df12_67', 'df12_567', 'df123_7', 'df123_67', 'df123_567', 'df1234_567',
              'df123_4567', 'df1_3_7', 'df12_34_67', 'df12_34_567', 'df123_4_567', 'df12_345_67', 'df123_45_67']
# To keep track of the best accuracy and the feature combination

i = -1
for df in dataframes:
    best_accuracy = 0
    best_combination = None
    i +=1
    # Try all combinations of features (from 1 to all features)
    for r in range(1, len(features) + 1):
        for combination in itertools.combinations(features, r):
            # Select features for the current combination
            X = df[list(combination)]  # Features (for all shapes)
            y = df['Session']  # Labels (Session numbers)

            # Split data into training (80%) and testing (20%)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Classifier: RandomForest
            clf = RandomForestClassifier(random_state=42)
            clf.fit(X_train, y_train)

            # Predictions
            y_pred = clf.predict(X_test)

            # Accuracy
            accuracy = accuracy_score(y_test, y_pred)

            # If current combination has a higher accuracy, update best result
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_combination = combination

    # Print the best result for the combined dataset
    print(f"df {dataframenames[i]},len {len(df)} Best Combination: {best_combination}, Accuracy: {best_accuracy:.2f}")


# %%
#subject based cv
import pandas as pd
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score

# List of features to use
features = ['Mean_Velocity', 'SSIM_Score', 'Smoothness_Value', 'Time_Value', 'Curvature', 'Acceleration', 'Jerk']

# Dataframes and their names
dataframes = [df1_6]
dataframenames = ['df1_2', 'df1_56', 'df1_6', 'df1_7', 'df1_67', 'df1_567', 'df12_7', 'df12_67', 'df12_567', 'df123_7', 'df123_67', 'df123_567', 'df1234_567',
                  'df123_4567', 'df1_3_7', 'df12_34_67', 'df12_34_567', 'df123_4_567', 'df12_345_67', 'df123_45_67']

i = -1
for df in dataframes:
    best_accuracy = 0
    best_combination = None
    i += 1
    # Group column (subject IDs)
    groups = df['Subject']  # Ensure 'Subject' column is in your dataframe

    # Try all combinations of features (from 1 to all features)
    for r in range(1, len(features) + 1):
        for combination in itertools.combinations(features, r):
            # Select features for the current combination
            X = df[list(combination)]  # Features (for all shapes)
            y = df['Session']  # Labels (Session numbers)

            # Leave-One-Subject-Out Cross-Validation
            logo = LeaveOneGroupOut()
            accuracies = []

            for train_idx, test_idx in logo.split(X, y, groups):
                # Split data into training and testing
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # Train the classifier
                clf = RandomForestClassifier(random_state=42)
                clf.fit(X_train, y_train)

                # Predict and evaluate
                y_pred = clf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                accuracies.append(acc)

            # Average accuracy across folds
            avg_accuracy = sum(accuracies) / len(accuracies)

            # If current combination has a higher accuracy, update best result
            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                best_combination = combination

    # Print the best result for the combined dataset
    print(f"df {dataframenames[i]}, len {len(df)} Best Combination: {best_combination}, Accuracy: {best_accuracy:.2f}")


# %%
import tensorflow as tf
with tf.device('/GPU:0'):
    # Place your model training code here
    #subject based cv
    import pandas as pd
    import itertools
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import accuracy_score

    # List of features to use
    features = ['Mean_Velocity', 'SSIM_Score', 'Smoothness_Value', 'Time_Value', 'Curvature', 'Acceleration', 'Jerk']

    # Dataframes and their names
    dataframes = [df1_6]
    dataframenames = ['df1_2', 'df1_56', 'df1_6', 'df1_7', 'df1_67', 'df1_567', 'df12_7', 'df12_67', 'df12_567', 'df123_7', 'df123_67', 'df123_567', 'df1234_567',
                      'df123_4567', 'df1_3_7', 'df12_34_67', 'df12_34_567', 'df123_4_567', 'df12_345_67', 'df123_45_67']

    i = -1
    for df in dataframes:
        best_accuracy = 0
        best_combination = None
        i += 1
        # Group column (subject IDs)
        groups = df['Subject']  # Ensure 'Subject' column is in your dataframe

        # Try all combinations of features (from 1 to all features)
        for r in range(1, len(features) + 1):
            for combination in itertools.combinations(features, r):
                # Select features for the current combination
                X = df[list(combination)]  # Features (for all shapes)
                y = df['Session']  # Labels (Session numbers)

                # Leave-One-Subject-Out Cross-Validation
                logo = LeaveOneGroupOut()
                accuracies = []

                for train_idx, test_idx in logo.split(X, y, groups):
                    # Split data into training and testing
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                    # Train the classifier
                    clf = RandomForestClassifier(random_state=42)
                    clf.fit(X_train, y_train)

                    # Predict and evaluate
                    y_pred = clf.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    accuracies.append(acc)

                # Average accuracy across folds
                avg_accuracy = sum(accuracies) / len(accuracies)

                # If current combination has a higher accuracy, update best result
                if avg_accuracy > best_accuracy:
                    best_accuracy = avg_accuracy
                    best_combination = combination

        # Print the best result for the combined dataset
        print(f"df {dataframenames[i]}, len {len(df)} Best Combination: {best_combination}, Accuracy: {best_accuracy:.2f}")


# %%


# %%
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import itertools

# List of features to use
features = ['Mean_Velocity', 'SSIM_Score', 'Smoothness_Value', 'Time_Value', 'Curvature', 'Acceleration', 'Jerk']

dataframes = [df1_2, df1_56, df1_6, df1_7, df1_67, df1_567, df12_7, df12_67, df12_567, df123_7, df123_67, df123_567,
              df1234_567, df123_4567, df1_3_7, df12_34_67, df12_34_567, df123_4_567, df12_345_67, df123_45_67]
dataframenames = ['df1_2', 'df1_56', 'df1_6', 'df1_7', 'df1_67', 'df1_567', 'df12_7', 'df12_67', 'df12_567',
                 'df123_7', 'df123_67', 'df123_567', 'df1234_567', 'df123_4567', 'df1_3_7', 'df12_34_67',
                 'df12_34_567', 'df123_4_567', 'df12_345_67', 'df123_45_67']

i = -1
for df in dataframes:
    best_accuracy = 0
    best_combination = None
    i += 1
    for r in range(1, len(features) + 1):
        for combination in itertools.combinations(features, r):
            X = df[list(combination)]
            y = df['Session']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Classifier: SVM
            clf = SVC(kernel='poly', random_state=42)  # You can change the kernel (e.g., 'rbf', 'poly')
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_combination = combination

    print(f"df {dataframenames[i]}, len {len(df)} Best Combination: {best_combination}, Accuracy: {best_accuracy:.2f}")

# %%
"""
from sklearn.model_selection import cross_val_score

# List of features to use
features = ['Mean_Velocity', 'SSIM_Score', 'Smoothness_Value', 'Time_Value', 'Curvature', 'Acceleration', 'Jerk']
dataframes = [df1_2, df1_56, df1_6, df1_7, df1_67, df1_567, df12_7, df12_67, df12_567, df123_7, df123_67, df123_567,
              df1234_567, df123_4567, df1_3_7, df12_34_67, df12_34_567, df123_4_567, df12_345_67, df123_45_67]
dataframenames = ['df1_2', 'df1_56', 'df1_6', 'df1_7', 'df1_67', 'df1_567', 'df12_7', 'df12_67', 'df12_567',
                 'df123_7', 'df123_67', 'df123_567', 'df1234_567', 'df123_4567', 'df1_3_7', 'df12_34_67',
                 'df12_34_567', 'df123_4_567', 'df12_345_67', 'df123_45_67']

i = -1
for df in dataframes:
    best_accuracy = 0
    best_combination = None
    i += 1
    for r in range(1, len(features) + 1):
        for combination in itertools.combinations(features, r):
            X = df[list(combination)]
            y = df['Session']

            # Classifier: SVM with Cross-validation
            clf = SVC(kernel='poly', random_state=42)
            cv_scores = cross_val_score(clf, X, y, cv=5)  # 5-fold cross-validation
            mean_accuracy = np.mean(cv_scores)

            if mean_accuracy > best_accuracy:
                best_accuracy = mean_accuracy
                best_combination = combination

    print(f"df {dataframenames[i]}, len {len(df)} Best Combination: {best_combination}, Mean CV Accuracy: {best_accuracy:.2f}")"""

# %%
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline

# List of features to use
features = ['Mean_Velocity', 'SSIM_Score', 'Smoothness_Value', 'Time_Value', 'Curvature', 'Acceleration', 'Jerk']

# Replace these with actual dataframes containing the corresponding data
dataframes = [df1_2, df1_56, df1_6, df1_7, df1_67, df1_567, df12_7, df12_67, df12_567, df123_7, df123_67, df123_567,
              df1234_567, df123_4567, df1_3_7, df12_34_67, df12_34_567, df123_4_567, df12_345_67, df123_45_67]
dataframenames = ['df1_2', 'df1_56', 'df1_6', 'df1_7', 'df1_67', 'df1_567', 'df12_7', 'df12_67', 'df12_567',
                 'df123_7', 'df123_67', 'df123_567', 'df1234_567', 'df123_4567', 'df1_3_7', 'df12_34_67',
                 'df12_34_567', 'df123_4_567', 'df12_345_67', 'df123_45_67']

i = -1
for df in dataframes:
    i += 1
    print(f"Evaluating {dataframenames[i]} with {len(df)} samples")

    # Separate features and target
    X = df[features]
    y = df['Session']

    # Feature Selection Techniques
    feature_selection_results = {}

    ## 1. SelectKBest (Filter Method) with best k selection
    # Create a pipeline with SelectKBest and SVC
    pipeline = Pipeline([
        ('select_k_best', SelectKBest(score_func=chi2)),
        ('svc', SVC(kernel='linear', random_state=42))  # Using linear kernel for speed
    ])

    # GridSearch for best 'k' in SelectKBest
    param_grid = {'select_k_best__k': np.arange(1, len(features) + 1)}
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)

    best_k = grid_search.best_params_['select_k_best__k']
    best_k_accuracy = grid_search.best_score_
    selected_features_kbest = np.array(features)[grid_search.best_estimator_.named_steps['select_k_best'].get_support()]
    feature_selection_results["SelectKBest"] = (selected_features_kbest, best_k_accuracy)

    ## 2. Recursive Feature Elimination (Wrapper Method)
    model = SVC(kernel='linear', random_state=42)  # Using linear SVC for compatibility with RFE
    selector_rfe = RFE(estimator=model, n_features_to_select=3)  # Select top 3 features
    X_rfe = selector_rfe.fit_transform(X, y)
    selected_features_rfe = np.array(features)[selector_rfe.get_support()]
    accuracy_rfe = np.mean(cross_val_score(model, X_rfe, y, cv=5))
    feature_selection_results["RFE"] = (selected_features_rfe, accuracy_rfe)

    ## 3. Lasso (Embedded Method)
    lasso = Lasso(alpha=0.01)
    lasso.fit(X, y)
    selected_features_lasso = np.array(features)[lasso.coef_ != 0]
    X_lasso = df[selected_features_lasso]
    accuracy_lasso = np.mean(cross_val_score(SVC(kernel='linear', random_state=42), X_lasso, y, cv=5))
    feature_selection_results["Lasso"] = (selected_features_lasso, accuracy_lasso)

    # Print results for each technique
    for method, (selected_feats, accuracy) in feature_selection_results.items():
        print(f"Method: {method}, Selected Features: {list(selected_feats)}, Mean CV Accuracy: {accuracy:.2f}")
    print("\n" + "-"*60 + "\n")


# %%
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.linear_model import Lasso
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

# List of features to use
features = ['Mean_Velocity', 'SSIM_Score', 'Smoothness_Value', 'Time_Value', 'Curvature', 'Acceleration', 'Jerk']

# Replace these with actual dataframes containing the corresponding data
dataframes = [df1_2, df1_56, df1_6, df1_7, df1_67, df1_567, df12_7, df12_67, df12_567, df123_7, df123_67, df123_567,
              df1234_567, df123_4567, df1_3_7, df12_34_67, df12_34_567, df123_4_567]
dataframenames = ['df1_2', 'df1_56', 'df1_6', 'df1_7', 'df1_67', 'df1_567', 'df12_7', 'df12_67', 'df12_567',
                 'df123_7', 'df123_67', 'df123_567', 'df1234_567', 'df123_4567', 'df1_3_7', 'df12_34_67',
                 'df12_34_567', 'df123_4_567']

i = -1
for df in dataframes:
    i += 1
    print(f"Evaluating {dataframenames[i]} with {len(df)} samples")

    # Separate features and target
    X = df[features]
    y = df['Session']
    groups = df['Subject']  # Assuming 'Subject' column exists in your dataframe

    # Initialize Leave-One-Subject-Out Cross-Validation
    logo = LeaveOneGroupOut()

    # Feature Selection Techniques
    feature_selection_results = {}

    ## 1. SelectKBest (Filter Method) with best k selection
    pipeline = Pipeline([
        ('select_k_best', SelectKBest(score_func=chi2)),
        ('svc', SVC(kernel='linear', random_state=42))  # Using linear kernel for speed
    ])

    param_grid = {'select_k_best__k': np.arange(1, len(features) + 1)}
    grid_search = GridSearchCV(pipeline, param_grid, cv=logo.split(X, y, groups), scoring='accuracy')
    grid_search.fit(X, y)

    best_k = grid_search.best_params_['select_k_best__k']
    best_k_accuracy = grid_search.best_score_
    selected_features_kbest = np.array(features)[grid_search.best_estimator_.named_steps['select_k_best'].get_support()]
    feature_selection_results["SelectKBest"] = (selected_features_kbest, best_k_accuracy)

    ## 2. Recursive Feature Elimination (Wrapper Method)
    model = SVC(kernel='linear', random_state=42)
    selector_rfe = RFE(estimator=model, n_features_to_select=3)
    X_rfe = selector_rfe.fit_transform(X, y)
    selected_features_rfe = np.array(features)[selector_rfe.get_support()]
    accuracy_rfe = np.mean(cross_val_score(model, X_rfe, y, cv=logo.split(X_rfe, y, groups)))
    feature_selection_results["RFE"] = (selected_features_rfe, accuracy_rfe)

    ## 3. Lasso (Embedded Method)
    lasso = Lasso(alpha=0.01)
    lasso.fit(X, y)
    selected_features_lasso = np.array(features)[lasso.coef_ != 0]
    X_lasso = df[selected_features_lasso]
    accuracy_lasso = np.mean(cross_val_score(SVC(kernel='linear', random_state=42), X_lasso, y, cv=logo.split(X_lasso, y, groups)))
    feature_selection_results["Lasso"] = (selected_features_lasso, accuracy_lasso)

    # Print results for each technique
    for method, (selected_feats, accuracy) in feature_selection_results.items():
        print(f"Method: {method}, Selected Features: {list(selected_feats)}, Mean LOSO Accuracy: {accuracy:.2f}")
    print("\n" + "-"*60 + "\n")


# %%
#each shape seperately:
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.linear_model import Lasso
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

# List of features to use
features = ['Mean_Velocity', 'SSIM_Score', 'Smoothness_Value', 'Time_Value', 'Curvature', 'Acceleration', 'Jerk']

# Assuming the dataset contains a 'Shape' column
dataframes = [your_combined_dataframe]  # Replace with the actual dataframe containing all shapes

for shape in dataframes[0]['Shape'].unique():
    print(f"\nProcessing for Shape: {shape}\n")
    df_shape = dataframes[0][dataframes[0]['Shape'] == shape]

    print(f"Evaluating Shape {shape} with {len(df_shape)} samples")

    # Separate features and target
    X = df_shape[features]
    y = df_shape['Session']
    groups = df_shape['Subject']  # Assuming 'Subject' column exists in your dataframe

    # Initialize Leave-One-Subject-Out Cross-Validation
    logo = LeaveOneGroupOut()

    # Feature Selection Techniques
    feature_selection_results = {}

    ## 1. SelectKBest (Filter Method) with best k selection
    pipeline = Pipeline([
        ('select_k_best', SelectKBest(score_func=chi2)),
        ('svc', SVC(kernel='linear', random_state=42))  # Using linear kernel for speed
    ])

    param_grid = {'select_k_best__k': np.arange(1, len(features) + 1)}
    grid_search = GridSearchCV(pipeline, param_grid, cv=logo.split(X, y, groups), scoring='accuracy')
    grid_search.fit(X, y)

    best_k = grid_search.best_params_['select_k_best__k']
    best_k_accuracy = grid_search.best_score_
    selected_features_kbest = np.array(features)[grid_search.best_estimator_.named_steps['select_k_best'].get_support()]
    feature_selection_results["SelectKBest"] = (selected_features_kbest, best_k_accuracy)

    ## 2. Recursive Feature Elimination (Wrapper Method)
    model = SVC(kernel='linear', random_state=42)
    selector_rfe = RFE(estimator=model, n_features_to_select=3)
    X_rfe = selector_rfe.fit_transform(X, y)
    selected_features_rfe = np.array(features)[selector_rfe.get_support()]
    accuracy_rfe = np.mean(cross_val_score(model, X_rfe, y, cv=logo.split(X_rfe, y, groups)))
    feature_selection_results["RFE"] = (selected_features_rfe, accuracy_rfe)

    ## 3. Lasso (Embedded Method)
    lasso = Lasso(alpha=0.01)
    lasso.fit(X, y)
    selected_features_lasso = np.array(features)[lasso.coef_ != 0]
    X_lasso = df_shape[selected_features_lasso]
    accuracy_lasso = np.mean(cross_val_score(SVC(kernel='linear', random_state=42), X_lasso, y, cv=logo.split(X_lasso, y, groups)))
    feature_selection_results["Lasso"] = (selected_features_lasso, accuracy_lasso)

    # Print results for each technique
    for method, (selected_feats, accuracy) in feature_selection_results.items():
        print(f"Method: {method}, Selected Features: {list(selected_feats)}, Mean LOSO Accuracy: {accuracy:.2f}")
    print("\n" + "-"*60 + "\n")

# %%
# List of features to use
features = ['Mean_Velocity', 'SSIM_Score', 'Smoothness_Value', 'Time_Value', 'Curvature', 'Acceleration', 'Jerk']
#features = [ 'Jerk', 'Curvature']

# Separate features and target
X = df1_6[features]
y = df1_6['Session']

# Separate data into training and testing sets for each shape
shapes = df1_6['Shape'].unique()
for shape in shapes:
  X_shape = X[df1_6['Shape'] == shape]
  y_shape = y[df1_6['Shape'] == shape]

  # Feature Selection Techniques
  feature_selection_results = {}

  ## 1. SelectKBest (Filter Method) with best k selection
  # Create a pipeline with SelectKBest and SVC
  pipeline = Pipeline([
      ('select_k_best', SelectKBest(score_func=chi2)),
      ('svc', SVC(kernel='rbf', random_state=42))  # Using linear kernel for speed
  ])

  # GridSearch for best 'k' in SelectKBest
  param_grid = {'select_k_best__k': np.arange(1, len(features) + 1)}
  grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
  grid_search.fit(X_shape, y_shape)

  best_k = grid_search.best_params_['select_k_best__k']
  best_k_accuracy = grid_search.best_score_
  selected_features_kbest = np.array(features)[grid_search.best_estimator_.named_steps['select_k_best'].get_support()]
  feature_selection_results["SelectKBest"] = (selected_features_kbest, best_k_accuracy)

  ## 2. Recursive Feature Elimination (Wrapper Method)
  model = SVC(kernel='linear', random_state=42)  # Using linear SVC for compatibility with RFE
  selector_rfe = RFE(estimator=model, n_features_to_select=3)  # Select top 3 features
  X_rfe = selector_rfe.fit_transform(X_shape, y_shape)
  selected_features_rfe = np.array(features)[selector_rfe.get_support()]
  accuracy_rfe = np.mean(cross_val_score(model, X_rfe, y_shape, cv=5))
  feature_selection_results["RFE"] = (selected_features_rfe, accuracy_rfe)

  ## 3. Lasso (Embedded Method)
  lasso = Lasso(alpha=0.01)
  lasso.fit(X_shape, y_shape)
  selected_features_lasso = np.array(features)[lasso.coef_ != 0]
  X_lasso = df1_6[selected_features_lasso][df1_6['Shape'] == shape]
  accuracy_lasso = np.mean(cross_val_score(SVC(kernel='linear', random_state=42), X_lasso, y_shape, cv=5))
  feature_selection_results["Lasso"] = (selected_features_lasso, accuracy_lasso)

  # Print results for each technique
  print(f"Shape: {shape}")
  for method, (selected_feats, accuracy) in feature_selection_results.items():
      print(f"Method: {method}, Selected Features: {list(selected_feats)}, Mean CV Accuracy: {accuracy:.2f}")
  print("\n" + "-"*60 + "\n")

# %%

# List of features to use
features = ['Mean_Velocity', 'SSIM_Score', 'Smoothness_Value', 'Time_Value', 'Curvature', 'Acceleration', 'Jerk']

# Separate features and target
X = df1_6[features]
y = df1_6['Session']

# Feature Selection Techniques
feature_selection_results = {}

## 1. SelectKBest (Filter Method) with best k selection
# Create a pipeline with SelectKBest and SVC
pipeline = Pipeline([
    ('select_k_best', SelectKBest(score_func=chi2)),
    ('svc', SVC(kernel='linear', random_state=42))  # Using linear kernel for speed
])

# GridSearch for best 'k' in SelectKBest
param_grid = {'select_k_best__k': np.arange(1, len(features) + 1)}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

best_k = grid_search.best_params_['select_k_best__k']
best_k_accuracy = grid_search.best_score_
selected_features_kbest = np.array(features)[grid_search.best_estimator_.named_steps['select_k_best'].get_support()]
feature_selection_results["SelectKBest"] = (selected_features_kbest, best_k_accuracy)

## 2. Recursive Feature Elimination (Wrapper Method)
model = SVC(kernel='linear', random_state=42)  # Using linear SVC for compatibility with RFE
selector_rfe = RFE(estimator=model, n_features_to_select=3)  # Select top 3 features
X_rfe = selector_rfe.fit_transform(X, y)
selected_features_rfe = np.array(features)[selector_rfe.get_support()]
accuracy_rfe = np.mean(cross_val_score(model, X_rfe, y, cv=5))
feature_selection_results["RFE"] = (selected_features_rfe, accuracy_rfe)

## 3. Lasso (Embedded Method)
lasso = Lasso(alpha=0.01)
lasso.fit(X, y)
selected_features_lasso = np.array(features)[lasso.coef_ != 0]
X_lasso = df1_6[selected_features_lasso]
accuracy_lasso = np.mean(cross_val_score(SVC(kernel='linear', random_state=42), X_lasso, y, cv=5))
feature_selection_results["Lasso"] = (selected_features_lasso, accuracy_lasso)

# Print results for each technique
for method, (selected_feats, accuracy) in feature_selection_results.items():
    print(f"Method: {method}, Selected Features: {list(selected_feats)}, Mean CV Accuracy: {accuracy:.2f}")

# %%
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score, StratifiedKFold

# ... (Your existing code) ...


# List of features to use
features = ['Mean_Velocity', 'SSIM_Score', 'Smoothness_Value', 'Time_Value', 'Curvature', 'Acceleration', 'Jerk']
dataframes = [df1_2, df1_56, df1_6, df1_7, df1_67, df1_567, df12_7, df12_67, df12_567, df123_7, df123_67, df123_567,
              df1234_567, df123_4567, df1_3_7, df12_34_67, df12_34_567, df123_4_567, df12_345_67, df123_45_67]
dataframenames = ['df1_2', 'df1_56', 'df1_6', 'df1_7', 'df1_67', 'df1_567', 'df12_7', 'df12_67', 'df12_567',
                 'df123_7', 'df123_67', 'df123_567', 'df1234_567', 'df123_4567', 'df1_3_7', 'df12_34_67',
                 'df12_34_567', 'df123_4_567', 'df12_345_67', 'df123_45_67']

i = -1
for df in dataframes:
    best_accuracy = 0
    best_k = None
    i += 1
    X = df[features]
    y = df['Session']

    # Feature selection using ANOVA F-value
    selector = SelectKBest(f_classif, k=3)  # You can change 'k' to select a different number of features
    X_new = selector.fit_transform(X, y)

    # Get the indices of the selected features
    selected_features_indices = selector.get_support(indices=True)
    selected_features = [features[i] for i in selected_features_indices]

    #print(selected_features)

    # Cross-validation with SVM
    clf = SVC(kernel='poly', random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_new, y, cv=cv, scoring='accuracy')
    accuracy = scores.mean()

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = selected_features

    print(f"df {dataframenames[i]}, len {len(df)} Best Features: {best_k}, Accuracy: {best_accuracy:.2f}")

# %%
from google.colab import drive
import scipy.io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from skimage.metrics import structural_similarity as ssim
import cv2
import itertools
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif


# ... (Your existing code for data loading and preprocessing) ...


# List of features to use
features = ['Mean_Velocity', 'SSIM_Score', 'Smoothness_Value', 'Time_Value', 'Curvature', 'Acceleration', 'Jerk']
dataframes = [df1_2, df1_56, df1_6, df1_7, df1_67, df1_567, df12_7, df12_67, df12_567, df123_7, df123_67, df123_567,
              df1234_567, df123_4567, df1_3_7, df12_34_67, df12_34_567, df123_4_567, df12_345_67, df123_45_67]
dataframenames = ['df1_2', 'df1_56', 'df1_6', 'df1_7', 'df1_67', 'df1_567', 'df12_7', 'df12_67', 'df12_567',
                 'df123_7', 'df123_67', 'df123_567', 'df1234_567', 'df123_4567', 'df1_3_7', 'df12_34_67',
                 'df12_34_567', 'df123_4_567', 'df12_345_67', 'df123_45_67']

i = -1
for df in dataframes:
    best_accuracy = 0
    best_combination = None
    i += 1

    # Iterate through each shape
    for shape in shapes:
        df_shape = df[df['Shape'] == shape]
        if not df_shape.empty:
            X = df_shape[features]
            y = df_shape['Session']

            # Feature Selection using ANOVA F-value (you can try other methods)
            selector = SelectKBest(f_classif, k=min(3, len(features))) # Select top 3 features
            X_new = selector.fit_transform(X, y)

            # Get the selected feature indices
            selected_feature_indices = selector.get_support(indices=True)
            selected_features = [features[i] for i in selected_feature_indices]

            # Split data into training and testing
            X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

            # Classifier: SVM with cross-validation
            clf = SVC(kernel='poly', random_state=42)
            cv_scores = cross_val_score(clf, X_train, y_train, cv=5)  # 5-fold cross-validation

            # Average accuracy from cross-validation
            mean_accuracy = np.mean(cv_scores)

            # Fit the model on the training set for testing later
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)

            # If current shape and feature combination has a higher accuracy, update best result
            if mean_accuracy > best_accuracy:
                best_accuracy = mean_accuracy
                best_combination = selected_features

            print(f"df {dataframenames[i]}, Shape: {shape}, Selected Features: {selected_features}, "
                  f"Cross-Validation Accuracy: {mean_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}")

    # Print the best result for the combined dataset
    print(f"df {dataframenames[i]}, len {len(df)} Best Combination: {best_combination}, "
          f"Best Cross-Validation Accuracy: {best_accuracy:.2f}")

# %%

# List of features to use
features = ['Mean_Velocity', 'SSIM_Score', 'Smoothness_Value', 'Time_Value', 'Curvature', 'Acceleration', 'Jerk']
dataframes = [df1_2, df1_56, df1_6, df1_7, df1_67, df1_567, df12_7, df12_67, df12_567, df123_7, df123_67, df123_567,
              df1234_567, df123_4567, df1_3_7, df12_34_67, df12_34_567, df123_4_567, df12_345_67, df123_45_67]
dataframenames = ['df1_2', 'df1_56', 'df1_6', 'df1_7', 'df1_67', 'df1_567', 'df12_7', 'df12_67', 'df12_567',
                 'df123_7', 'df123_67', 'df123_567', 'df1234_567', 'df123_4567', 'df1_3_7', 'df12_34_67',
                 'df12_34_567', 'df123_4_567', 'df12_345_67', 'df123_45_67']

i = -1
for df in dataframes:
    i += 1
    print()
    print()
    print('in dataframe:', dataframenames[i])
    for shape in df['Shape'].unique():  # Iterate through each unique shape in the DataFrame
        best_cv_score = 0
        best_combination = None
        df_shape = df[df['Shape'] == shape]  # Filter the DataFrame for the current shape

        for r in range(1, len(features) + 1):
            for combination in itertools.combinations(features, r):
                X = df_shape[list(combination)]
                y = df_shape['Session']

                # Classifier: SVM
                clf = SVC(kernel='poly', random_state=42)

                # Split data into training (80%) and testing (20%)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train the classifier
                clf.fit(X_train, y_train)

                # Make predictions on the test set
                y_pred = clf.predict(X_test)

                # Calculate accuracy
                accuracy = accuracy_score(y_test, y_pred)

                if accuracy > best_cv_score:
                    best_cv_score = accuracy
                    best_combination = combination


        print(f"df {dataframenames[i]}, Shape: {shape}, Best Combination: {best_combination}, Accuracy: {best_cv_score:.2f}")

# %%

# List of features to use
features = ['Mean_Velocity', 'SSIM_Score', 'Smoothness_Value', 'Time_Value', 'Curvature', 'Acceleration', 'Jerk']
dataframes = [df1_2, df1_56, df1_6, df1_7, df1_67, df1_567, df12_7, df12_67, df12_567, df123_7, df123_67, df123_567,
              df1234_567, df123_4567, df1_3_7, df12_34_67, df12_34_567, df123_4_567, df12_345_67, df123_45_67]
dataframenames = ['df1_2', 'df1_56', 'df1_6', 'df1_7', 'df1_67', 'df1_567', 'df12_7', 'df12_67', 'df12_567',
                 'df123_7', 'df123_67', 'df123_567', 'df1234_567', 'df123_4567', 'df1_3_7', 'df12_34_67',
                 'df12_34_567', 'df123_4_567', 'df12_345_67', 'df123_45_67']

i = -1
for df in dataframes:
    i += 1
    print()
    print()
    print('in dataframe:', dataframenames[i])
    for shape in df['Shape'].unique():  # Iterate through each unique shape in the DataFrame
        best_accuracy = 0
        best_combination = None
        df_shape = df[df['Shape'] == shape]  # Filter the DataFrame for the current shape

        from sklearn.feature_selection import SelectKBest, f_classif

        for r in range(1, min(len(features), len(df_shape)) + 1):
          # Use SelectKBest for feature selection
          selector = SelectKBest(f_classif, k=r)
          X = df_shape[features]
          y = df_shape['Session']

          X_new = selector.fit_transform(X, y)

          selected_features = [features[i] for i in selector.get_support(indices=True)]
          X = df_shape[selected_features]

          # Split data into training (80%) and testing (20%)
          X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

          # Classifier: SVM
          clf = SVC(kernel='poly', random_state=42)

          # Train the classifier
          clf.fit(X_train, y_train)

          # Make predictions on the test set
          y_pred = clf.predict(X_test)

          # Calculate accuracy
          accuracy = accuracy_score(y_test, y_pred)

          if accuracy > best_accuracy:
              best_accuracy = accuracy
              best_combination = selected_features

        print(f"df {dataframenames[i]}, Shape: {shape}, Best Combination: {best_combination}, Accuracy: {best_accuracy:.2f}")

# %%
#cross val
import pandas as pd
import itertools
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# List of features to use
features = ['Mean_Velocity', 'SSIM_Score', 'Smoothness_Value', 'Time_Value', 'Curvature', 'Acceleration', 'Jerk']
dataframes = [df1_2, df1_56, df1_6, df1_7, df1_67, df1_567, df12_7, df12_67, df12_567, df123_7, df123_67, df123_567, df1234_567,
              df123_4567, df1_3_7, df12_34_67, df12_34_567, df123_4_567, df12_345_67, df123_45_67]
dataframenames =['df1_2', 'df1_56', 'df1_6','df1_7', 'df1_67', 'df1_567', 'df12_7', 'df12_67', 'df12_567', 'df123_7', 'df123_67', 'df123_567', 'df1234_567',
              'df123_4567', 'df1_3_7', 'df12_34_67', 'df12_34_567', 'df123_4_567', 'df12_345_67', 'df123_45_67']

# To keep track of the best accuracy and the feature combination

i = -1

# Iterate over each dataframe
for df in dataframes:
    best_accuracy = 0
    best_combination = None
    i += 1
    # Try all combinations of features (from 1 to all features)
    for r in range(1, len(features) + 1):
        for combination in itertools.combinations(features, r):
            # Select features for the current combination
            X = df[list(combination)]  # Features (for all shapes)
            y = df['Session']  # Labels (Session numbers)

            # Classifier: RandomForest
            clf = RandomForestClassifier(random_state=42)

            # Perform cross-validation with 5 folds to ensure robust accuracy estimates
            scores = cross_val_score(clf, X, y, cv=5)

            # Calculate the mean accuracy across the 5 folds
            accuracy = scores.mean()

            # If current combination has a higher accuracy, update the best result
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_combination = combination

    # Print the best result for the current dataset
    print(f"df {dataframenames[i]}, len {len(df)} Best Combination: {best_combination}, Accuracy: {best_accuracy:.2f}")


# %%
##classification combination version
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Assume df is your DataFrame
shapes = ['B', 'M', 'O', 'Star', 'Clover']

# List of features to use
features = ['SSIM_Score', 'Smoothness_Value', 'Time_Value', 'Mean_Velocity', 'Curvature', 'Acceleration', 'Jerk']

dataframes = [ df1_7 ]

dataframenames =['df1_7']

# Dictionary to store the best results for each shape
best_results = {}
for df in dataframes:
      print()
      print()
      print('in dataframe:', dataframenames[dataframes.index(df)])
      for shape in shapes:
          shape_df = df[df['Shape'] == shape]

          # To keep track of the best accuracy and the feature combination
          best_accuracy = 0
          best_combination = None

          # Try all combinations of features (from 1 to all features)
          for r in range(1, len(features) + 1):
              for combination in itertools.combinations(features, r):
                  # Select features for the current combination
                  X = shape_df[list(combination)]  # Features
                  y = shape_df['Session']  # Labels (Session numbers)

                  # Split data into training (80%) and testing (20%)
                  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                  # Classifier: RandomForest
                  clf = RandomForestClassifier(random_state=42)
                  clf.fit(X_train, y_train)

                  # Predictions
                  y_pred = clf.predict(X_test)

                  # Accuracy
                  accuracy = accuracy_score(y_test, y_pred)
                  print(shape , list(combination), accuracy)

                  # If current combination has a higher accuracy, update best result
                  if accuracy > best_accuracy:
                      best_accuracy = accuracy
                      best_combination = combination
                  if accuracy == 1:
                    print('******')
                    #break

          # Store the best result for the current shape
          best_results[shape] = {
              'best_combination': best_combination,
              'best_accuracy': best_accuracy
          }

      # Print the best result for each shape
      for shape, result in best_results.items():
          print(f"Shape: {shape}, Best Combination: {result['best_combination']}, Accuracy: {result['best_accuracy']:.2f}")


# %%
##classification combination version
import pandas as pd
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# Assume df is your DataFrame
shapes = ['B', 'M', 'O', 'Star', 'Clover']

# List of features to use
features = ['SSIM_Score', 'Smoothness_Value', 'Time_Value', 'Mean_Velocity', 'Curvature', 'Acceleration', 'Jerk']

dataframes = [df1_56, df1_6, df1_7, df1_67, df1_567, df12_7, df12_67, df12_567, df123_7, df123_67, df123_567, df1234_567,
              df123_4567, df1_3_7, df12_34_67, df12_34_567, df123_4_567, df12_345_67, df123_45_67]
dataframenames =['df1_56', 'df1_6','df1_7', 'df1_67', 'df1_567', 'df12_7', 'df12_67', 'df12_567', 'df123_7', 'df123_67', 'df123_567', 'df1234_567',
              'df123_4567', 'df1_3_7', 'df12_34_67', 'df12_34_567', 'df123_4_567', 'df12_345_67', 'df123_45_67']

# Dictionary to store the best results for each shape
best_results = {}
i = -1
for df in dataframes:
    i+=1
    print()
    print()
    print('In dataframe:', dataframenames[i])

    for shape in shapes:
        shape_df = df[df['Shape'] == shape]

        # To keep track of the best accuracy and the feature combination
        best_accuracy = 0
        best_combination = None

        # Try all combinations of features (from 1 to all features)
        for r in range(1, len(features) + 1):
            for combination in itertools.combinations(features, r):
                # Select features for the current combination
                X = shape_df[list(combination)]  # Features
                y = shape_df['Session']  # Labels (Session numbers)

                # Classifier: RandomForest
                clf = RandomForestClassifier(random_state=42)

                # Perform 5-fold cross-validation
                cv_scores = cross_val_score(clf, X, y, cv=5)

                # Average accuracy across the 5 folds
                accuracy = cv_scores.mean()

                #print(f"Shape: {shape}, Features: {list(combination)}, Accuracy: {accuracy:.2f}")

                # If current combination has a higher accuracy, update best result
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_combination = combination

        # Store the best result for the current shape
        best_results[shape] = {
            'best_combination': best_combination,
            'best_accuracy': best_accuracy
        }

    # Print the best result for each shape
    for shape, result in best_results.items():
        print(f"Shape: {shape}, Best Combination: {result['best_combination']}, Accuracy: {result['best_accuracy']:.2f}")


# %%
#classification

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

shapes = ['B', 'M', 'O', 'Star', 'Clover']
for shape in shapes:
      shape_df = df[df['Shape'] == shape]
      for feature in ['SSIM_Score', 'Smoothness_Value', 'Time_Value', 'Mean_Velocity', 'Curvature']:
            X = shape_df[[feature]]  # Features
            y = shape_df['Session']  # Labels (Session numbers)

            # Split data into training (80%) and testing (20%)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            """print(len(X_train))
            print(len(X_test))"""

            # Classifier: RandomForest
            clf = RandomForestClassifier(random_state=42)
            clf.fit(X_train, y_train)

            # Predictions
            y_pred = clf.predict(X_test)

            # Accuracy
            accuracy = accuracy_score(y_test, y_pred)
            print('Shape:', shape, 'feature:', feature, f'Accuracy: {accuracy:.2f}')

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

shapes = ['B', 'M', 'O', 'Star', 'Clover']
for shape in shapes:
      shape_df = df[df['Shape'] == shape]
      for feature in ['SSIM_Score', 'Smoothness_Value', 'Time_Value', 'Mean_Velocity']:
            X = shape_df[[feature]]  # Features
            y = shape_df['Session']  # Labels (Session numbers)

            # Split data into training (80%) and testing (20%)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
            """print(len(X_train))
            print(len(X_test))"""

            # Classifier: RandomForest
            clf = RandomForestClassifier(random_state=42)
            clf.fit(X_train, y_train)

            # Predictions
            y_pred = clf.predict(X_test)

            # Accuracy
            accuracy = accuracy_score(y_test, y_pred)
            print('Shape:', shape, 'feature:', feature, f'Accuracy: {accuracy:.2f}')

# %% [markdown]
# # Clustering

# %%

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA

# Loop through each shape
for first_feature in ['SSIM_Score', 'Smoothness_Value', 'Time_Value', 'Mean_Velocity', 'Curvature', 'Acceleration', 'Jerk']:
  for second_feature in ['SSIM_Score', 'Smoothness_Value', 'Time_Value', 'Mean_Velocity', 'Curvature', 'Acceleration', 'Jerk']:
      shape = 'O'


      # Filter the DataFrame for the current shape
      shape_df = df[df['Shape'] == shape]
      shape_df['Time_Value']= 25-shape_df['Time_Value']

      # Extract the SSIM score and Time value as features
      X = shape_df[['SSIM_Score', 'Time_Value', 'Mean_Velocity', 'Smoothness_Value', 'Curvature']].dropna()



      kpca = KernelPCA(kernel='rbf', n_components=2)
      transformed_data = kpca.fit_transform(X)


      # Perform K-means clustering with 2 clusters
      kmeans = KMeans(n_clusters=2, random_state=0)
      kmeans.fit(transformed_data)

      # Add cluster labels to the DataFrame
      shape_df['Cluster'] = kmeans.labels_

      # Plot the clusters
      plt.figure()
      plt.scatter(shape_df[first_feature], shape_df[second_feature], c=shape_df['Session'], cmap='viridis')
      scatter = plt.scatter(shape_df[first_feature], shape_df[second_feature],c=shape_df['Session'], cmap='viridis')
      for i in range(len(shape_df)):
        plt.text(shape_df[first_feature].iloc[i],shape_df[second_feature].iloc[i], str(shape_df['Session'].iloc[i]), fontsize=9, ha='right')

      #plt.colorbar(scatter, label='Session')
      plt.xlabel(first_feature)
      plt.ylabel(second_feature)
      plt.title(f'Shape {shape}')
      plt.show()


# %%

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA

# Loop through each shape
for shape in shapes:
  # Filter the DataFrame for the current shape
  shape_df = df[df['Shape'] == shape]

  # Extract the SSIM score and Time value as features
  X = shape_df[['SSIM_Score', 'Time_Value', 'Mean_Velocity', 'Smoothness_Value', 'Curvature', 'Acceleration', 'Jerk']].dropna()



  kpca = KernelPCA(kernel='rbf', n_components=2)
  transformed_data = kpca.fit_transform(X)


  # Perform K-means clustering with 2 clusters
  kmeans = KMeans(n_clusters=2, random_state=0)
  kmeans.fit(transformed_data)

  # Add cluster labels to the DataFrame
  shape_df['Cluster'] = kmeans.labels_

  # Plot the clusters
  plt.figure()
  plt.scatter(shape_df['SSIM_Score'], shape_df['Smoothness_Value'], c=shape_df['Session'], cmap='viridis')
  scatter = plt.scatter(shape_df['SSIM_Score'], shape_df['Smoothness_Value'],c=shape_df['Session'], cmap='viridis')
  for i in range(len(shape_df)):
    plt.text(shape_df['SSIM_Score'].iloc[i],
             shape_df['Smoothness_Value'].iloc[i],
             str(shape_df['Session'].iloc[i]),
             fontsize=9, ha='right')

  plt.colorbar(scatter, label='Session')
  plt.xlabel('SSIM Score')
  plt.ylabel('Time Value')
  plt.title(f'K-Means Clustering for {shape}')
  plt.show()


# %%
#Smoothness

import pandas as pd
import matplotlib.pyplot as plt

plot_subject_session_files = range(6, 3)
plt.figure(figsize=(8, 5))
for shape in ['B', 'M', 'O', 'Star', 'Clover']:
    mean_times = []
    for session in range(1,8):
        session_mean_times = []
        for subject in range(1, 13):
                  row  = finaldf[(finaldf['subject'] == subject) &(finaldf['session'] == session) & (finaldf['shape'] == shape)]
                  if not row.empty:
                    time_value = row['mean_V'].values[0]
                    session_mean_times.append(time_value)
            # Calculate the mean and std deviation across trials for the session
        mean_times.append(np.nanmean(session_mean_times))

        # Plot with error bars

    plt.plot(sessions, mean_times, label=shape, marker='o', linestyle='--')

plt.title('Mean smoothness value Across Sessions for Different Shapes')
plt.xlabel('Sessions')
plt.ylabel('Mean dmoothness Value')
plt.legend(title="Shapes")
plt.show()



# %%
#kmeans real version
"""Best feature combinations for each shape:
B: SSIM_Score with accuracy 0.83
M: Smoothness_Value, Curvature with accuracy 0.79
O: SSIM_Score with accuracy 0.79
Star: Time_Value with accuracy 0.83
Clover: Time_Value with accuracy 0.83"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA

['Shape', 'Subject', 'Session', 'SSIM_Score','Smoothness_Value','Time_Value' ,'Mean_Velocity',  'Mean_v', 'Max_v', 'Std_v', 'Curvature','Acceleration', 'Jerk']
shapes = ['B', 'M', 'O', 'Star', 'Clover']
# Loop through each shape
for shape in shapes:
    # Filter the DataFrame for the current shape
    shape_df = df[df['Shape'] == shape]
    shape_df = df.copy()
    #shape_df2 = shape_df.copy()

    # Extract the SSIM score and Time value as features
    X = shape_df[['SSIM_Score','Smoothness_Value','Time_Value' ,'Mean_Velocity',  'Mean_v', 'Max_v', 'Std_v', 'Curvature','Acceleration', 'Jerk']]
    """if shape == 'B':
      X = shape_df[['SSIM_Score']]
    if shape == 'O':
      X = shape_df[['SSIM_Score']]
    if shape == 'Star':
      X = shape_df[['Time_Value']]
    if shape == 'Clover':
      X = shape_df[['Time_Value']]
    if shape == 'M':
      X = shape_df[['Smoothness_Value', 'Curvature']]"""


    if 'Time_Value' in X.columns:
        X['Time_Value'] = 25 - X['Time_Value']


    #x2 = shape_df[['Smoothness_Value', 'Time_Value']]

    # Perform K-means clustering with 3 clusters
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(X)
    #kmeans2 = KMeans(n_clusters=2, random_state=0)
    #kmeans2.fit(x2)

    # Add cluster labels to the DataFrame
    shape_df['Cluster'] = kmeans.labels_
    #shape_df2['Cluster'] = kmeans2.labels_

    #print(shape_df)

    # Plot the distribution of session numbers in each cluster
    fig, axx = plt.subplots(1, 1)

    # Set bar width for clusters
    bar_width = 0.1
    session_labels = sorted(shape_df['Session'].unique())  # Get sorted session numbers
    indices = np.arange(len(session_labels))  # X positions for the bars

    for cluster in range(2):
        cluster_df = shape_df[shape_df['Cluster'] == cluster]
        #cluster_df2 = shape_df2[shape_df2['Cluster'] == cluster]
        session_counts = cluster_df['Session'].value_counts().reindex(session_labels, fill_value=0)
        #session_counts2 = cluster_df2['Session'].value_counts().reindex(session_labels, fill_value=0)

        # Plot the bars for each cluster with an offset
        axx.bar(indices + bar_width * cluster, session_counts.values, bar_width, label=f'Cluster {cluster}')
        #axx[1].bar(indices + bar_width * cluster, session_counts2.values, bar_width, label=f'Cluster {cluster}')


    # Set x-ticks to be centered between the grouped bars
    axx.set_xticks(indices + bar_width / 2)
    axx.set_xticklabels(session_labels)
    #axx[1].set_xticks(indices + bar_width / 2)
    #axx[1].set_xticklabels(session_labels)

    plt.xlabel('Session Number')
    plt.ylabel('Number of Subjects')
    plt.title(f'Distribution of Session Numbers in Each Cluster for {shape}')
    plt.legend()
    plt.show()


# %%
#accuracy kmeans version
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment

# Function to match the cluster labels with the session labels
def match_labels(true_labels, predicted_labels):
    D = max(predicted_labels.max(), true_labels.max()) + 1
    cost_matrix = np.zeros((D, D), dtype=int)
    for i in range(len(true_labels)):
        cost_matrix[true_labels[i], predicted_labels[i]] += 1
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    new_predicted_labels = np.zeros_like(predicted_labels)
    for i in range(D):
        new_predicted_labels[predicted_labels == col_ind[i]] = row_ind[i]
    return new_predicted_labels

# Loop through each shape
shapes = ['B', 'M', 'O', 'Star', 'Clover']
for shape in shapes:
    # Filter the DataFrame for the current shape
    shape_df = df[df['Shape'] == shape]
    shape_df2 = shape_df.copy()

    # Extract features
    X = shape_df[['Time_Value', 'SSIM_Score', 'Smoothness_Value', 'Mean_Velocity']]
    X['Time_Value'] = X['Time_Value'] * -1

    x2 = shape_df[['Smoothness_Value', 'Time_Value']]

    # Perform K-means clustering with 2 clusters
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(X)
    kmeans2 = KMeans(n_clusters=2, random_state=0)
    kmeans2.fit(x2)

    # Add cluster labels to the DataFrame
    shape_df['Cluster'] = kmeans.labels_
    shape_df2['Cluster'] = kmeans2.labels_

    # True labels (Session as labels)
    true_labels = shape_df['Session'].values

    # Predicted labels from KMeans clustering
    predicted_labels = kmeans.labels_

    # Match the predicted labels to true labels
    matched_labels = match_labels(true_labels, predicted_labels)

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, matched_labels)
    print(f"Accuracy for {shape}: {accuracy:.2f}")

    # Plot the distribution of session numbers in each cluster
    fig, axx = plt.subplots(1, 1)

    # Set bar width for clusters
    bar_width = 0.1
    session_labels = sorted(shape_df['Session'].unique())  # Get sorted session numbers
    indices = np.arange(len(session_labels))  # X positions for the bars

    for cluster in range(2):
        cluster_df = shape_df[shape_df['Cluster'] == cluster]
        #cluster_df2 = shape_df2[shape_df2['Cluster'] == cluster]
        session_counts = cluster_df['Session'].value_counts().reindex(session_labels, fill_value=0)
        #session_counts2 = cluster_df2['Session'].value_counts().reindex(session_labels, fill_value=0)

        # Plot the bars for each cluster with an offset
        axx.bar(indices + bar_width * cluster, session_counts.values, bar_width, label=f'Cluster {cluster}')
        #axx[1].bar(indices + bar_width * cluster, session_counts2.values, bar_width, label=f'Cluster {cluster}')

    # Set x-ticks to be centered between the grouped bars
    axx.set_xticks(indices + bar_width / 2)
    axx.set_xticklabels(session_labels)
    """axx[1].set_xticks(indices + bar_width / 2)
    axx[1].set_xticklabels(session_labels)"""

    plt.xlabel('Session Number')
    plt.ylabel('Number of Subjects')
    plt.title(f'Distribution of Session Numbers in Each Cluster for {shape}')
    plt.legend()
    plt.show()


# %%
#kmeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
from itertools import combinations

# Function to match the cluster labels with the session labels
def match_labels(true_labels, predicted_labels):
    D = max(predicted_labels.max(), true_labels.max()) + 1
    cost_matrix = np.zeros((D, D), dtype=int)
    for i in range(len(true_labels)):
        cost_matrix[true_labels[i], predicted_labels[i]] += 1
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    new_predicted_labels = np.zeros_like(predicted_labels)
    for i in range(D):
        new_predicted_labels[predicted_labels == col_ind[i]] = row_ind[i]
    return new_predicted_labels

# Loop through each shape
shapes = ['B', 'M', 'O', 'Star', 'Clover']
features = ['Smoothness_Value', 'Mean_Velocity','Time_Value', 'SSIM_Score', 'Curvature', 'Acceleration', 'Jerk']

# Store the accuracy for each shape and feature combination
accuracy_results = {}

for shape in shapes:
    # Filter the DataFrame for the current shape and create a copy
    shape_df = df[df['Shape'] == shape].copy()
    shape_df = df.copy()

    # Get the true session labels
    true_labels = shape_df['Session'].values

    # Loop through all feature combinations
    for num_features in range(1, len(features) + 1):  # Try 2 to all features
        for feature_combination in combinations(features, num_features):
            # Extract the selected feature combination and create a copy
            X = shape_df[list(feature_combination)].copy()

            # Negate 'Time_Value' if it's in the selected features
            if 'Time_Value' in X.columns:
                X['Time_Value'] = X['Time_Value'] * -1  # Optional negation

            # Perform K-means clustering with 2 clusters
            kmeans = KMeans(n_clusters=2, random_state=0)
            kmeans.fit(X)

            # Add cluster labels to the DataFrame
            shape_df['Cluster'] = kmeans.labels_

            # Predicted labels from KMeans clustering
            predicted_labels = kmeans.labels_

            # Match the predicted labels to true labels
            matched_labels = match_labels(true_labels, predicted_labels)

            # Calculate accuracy
            accuracy = accuracy_score(true_labels, matched_labels)
            feature_combo_str = ', '.join(feature_combination)
            print(f"Accuracy for {shape} with features {feature_combo_str}: {accuracy:.2f}")

            # Store the accuracy result
            accuracy_results[(shape, feature_combo_str)] = accuracy

            # Plot the distribution of session numbers in each cluster
            fig, axx = plt.subplots(1, 1)

            # Set bar width for clusters
            bar_width = 0.1
            session_labels = sorted(shape_df['Session'].unique())  # Get sorted session numbers
            indices = np.arange(len(session_labels))  # X positions for the bars

            for cluster in range(2):
                cluster_df = shape_df[shape_df['Cluster'] == cluster]
                session_counts = cluster_df['Session'].value_counts().reindex(session_labels, fill_value=0)

                # Plot the bars for each cluster with an offset
                axx.bar(indices + bar_width * cluster, session_counts.values, bar_width, label=f'Cluster {cluster}')

            # Set x-ticks to be centered between the grouped bars
            axx.set_xticks(indices + bar_width / 2)
            axx.set_xticklabels(session_labels)

            plt.xlabel('Session Number')
            plt.ylabel('Number of Subjects')
            plt.title(f'Distribution of Session Numbers in Each Cluster for {shape} (Features: {feature_combo_str})')
            plt.legend()
            #plt.show()

# Print or analyze the best feature combinations for each shape
best_combinations = {shape: max([key for key in accuracy_results if key[0] == shape], key=lambda x: accuracy_results[x]) for shape in shapes}
print("Best feature combinations for each shape:")
for shape, best in best_combinations.items():
    print(f"{shape}: {best[1]} with accuracy {accuracy_results[best]:.2f}")


# %%
#KPCA --> K-means
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
from itertools import combinations

# Function to match the cluster labels with the session labels
def match_labels(true_labels, predicted_labels):
    D = max(predicted_labels.max(), true_labels.max()) + 1
    cost_matrix = np.zeros((D, D), dtype=int)
    for i in range(len(true_labels)):
        cost_matrix[true_labels[i], predicted_labels[i]] += 1
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    new_predicted_labels = np.zeros_like(predicted_labels)
    for i in range(D):
        new_predicted_labels[predicted_labels == col_ind[i]] = row_ind[i]
    return new_predicted_labels

# Loop through each shape
shapes = ['B', 'M', 'O', 'Star', 'Clover']
features = ['Smoothness_Value', 'Mean_Velocity','Time_Value', 'SSIM_Score', 'Curvature', 'Acceleration', 'Jerk']

# Store the accuracy for each shape and feature combination
accuracy_results = {}

for shape in shapes:
    # Filter the DataFrame for the current shape and create a copy
    shape_df = df[df['Shape'] == shape].copy()
    shape_df = df.copy()

    # Get the true session labels
    true_labels = shape_df['Session'].values

    # Loop through all feature combinations
    for num_features in range(1, len(features) + 1):  # Try 2 to all features
        for feature_combination in combinations(features, num_features):
            # Extract the selected feature combination and create a copy
            X = shape_df[list(feature_combination)].copy()

            # Negate 'Time_Value' if it's in the selected features
            if 'Time_Value' in X.columns:
                X['Time_Value'] = X['Time_Value'] * -1  # Optional negation

            # Perform K-means clustering with 2 clusters

            kpca = KernelPCA(kernel='rbf', n_components=2)
            transformed_data = kpca.fit_transform(X)


            # Perform K-means clustering with 2 clusters
            kmeans = KMeans(n_clusters=2, random_state=0)
            kmeans.fit(transformed_data)


            # Add cluster labels to the DataFrame
            shape_df['Cluster'] = kmeans.labels_

            # Predicted labels from KMeans clustering
            predicted_labels = kmeans.labels_

            # Match the predicted labels to true labels
            matched_labels = match_labels(true_labels, predicted_labels)

            # Calculate accuracy
            accuracy = accuracy_score(true_labels, matched_labels)
            feature_combo_str = ', '.join(feature_combination)
            print(f"Accuracy for {shape} with features {feature_combo_str}: {accuracy:.2f}")

            # Store the accuracy result
            accuracy_results[(shape, feature_combo_str)] = accuracy

            # Plot the distribution of session numbers in each cluster
            fig, axx = plt.subplots(1, 1)

            # Set bar width for clusters
            bar_width = 0.1
            session_labels = sorted(shape_df['Session'].unique())  # Get sorted session numbers
            indices = np.arange(len(session_labels))  # X positions for the bars

            for cluster in range(2):
                cluster_df = shape_df[shape_df['Cluster'] == cluster]
                session_counts = cluster_df['Session'].value_counts().reindex(session_labels, fill_value=0)

                # Plot the bars for each cluster with an offset
                axx.bar(indices + bar_width * cluster, session_counts.values, bar_width, label=f'Cluster {cluster}')

            # Set x-ticks to be centered between the grouped bars
            axx.set_xticks(indices + bar_width / 2)
            axx.set_xticklabels(session_labels)

            plt.xlabel('Session Number')
            plt.ylabel('Number of Subjects')
            plt.title(f'Distribution of Session Numbers in Each Cluster for {shape} (Features: {feature_combo_str})')
            plt.legend()
            #plt.show()

# Print or analyze the best feature combinations for each shape
best_combinations = {shape: max([key for key in accuracy_results if key[0] == shape], key=lambda x: accuracy_results[x]) for shape in shapes}
print("Best feature combinations for each shape:")
for shape, best in best_combinations.items():
    print(f"{shape}: {best[1]} with accuracy {accuracy_results[best]:.2f}")


# %%
#GMM

from sklearn.mixture import GaussianMixture

# Function to match the cluster labels with the session labels
def match_labels(true_labels, predicted_labels):
    D = max(predicted_labels.max(), true_labels.max()) + 1
    cost_matrix = np.zeros((D, D), dtype=int)
    for i in range(len(true_labels)):
        cost_matrix[true_labels[i], predicted_labels[i]] += 1
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    new_predicted_labels = np.zeros_like(predicted_labels)
    for i in range(D):
        new_predicted_labels[predicted_labels == col_ind[i]] = row_ind[i]
    return new_predicted_labels

# Loop through each shape
shapes = ['B', 'M', 'O', 'Star', 'Clover']
features = ['Smoothness_Value', 'Mean_Velocity','Time_Value', 'SSIM_Score', 'Curvature', 'Acceleration', 'Jerk']

# Store the accuracy for each shape and feature combination
accuracy_results = {}

for shape in shapes:
    # Filter the DataFrame for the current shape and create a copy
    shape_df = df[df['Shape'] == shape].copy()
    shape_df = df.copy()

    # Get the true session labels
    true_labels = shape_df['Session'].values

    # Loop through all feature combinations
    for num_features in range(1, len(features) + 1):  # Try 2 to all features
        for feature_combination in combinations(features, num_features):
            # Extract the selected feature combination and create a copy
            X = shape_df[list(feature_combination)].copy()

            # Negate 'Time_Value' if it's in the selected features
            if 'Time_Value' in X.columns:
                X['Time_Value'] = X['Time_Value'] * -1  # Optional negation

            # Perform GMM clustering with 2 components

            gmm = GaussianMixture(n_components=2, random_state=0)
            gmm.fit(X)

            # Add cluster labels to the DataFrame
            shape_df['Cluster'] = gmm.predict(X)

            # Predicted labels from GMM clustering
            predicted_labels = shape_df['Cluster'].values

            # Match the predicted labels to true labels
            matched_labels = match_labels(true_labels, predicted_labels)

            # Calculate accuracy
            accuracy = accuracy_score(true_labels, matched_labels)
            feature_combo_str = ', '.join(feature_combination)
            print(f"Accuracy for {shape} with features {feature_combo_str}: {accuracy:.2f}")

            # Store the accuracy result
            accuracy_results[(shape, feature_combo_str)] = accuracy

            # Plot the distribution of session numbers in each cluster
            fig, axx = plt.subplots(1, 1)

            # Set bar width for clusters
            bar_width = 0.1
            session_labels = sorted(shape_df['Session'].unique())  # Get sorted session numbers
            indices = np.arange(len(session_labels))  # X positions for the bars

            for cluster in range(2):
                cluster_df = shape_df[shape_df['Cluster'] == cluster]
                session_counts = cluster_df['Session'].value_counts().reindex(session_labels, fill_value=0)

                # Plot the bars for each cluster with an offset
                axx.bar(indices + bar_width * cluster, session_counts.values, bar_width, label=f'Cluster {cluster}')

            # Set x-ticks to be centered between the grouped bars
            axx.set_xticks(indices + bar_width / 2)
            axx.set_xticklabels(session_labels)

            plt.xlabel('Session Number')
            plt.ylabel('Number of Subjects')
            plt.title(f'Distribution of Session Numbers in Each Cluster for {shape} (Features: {feature_combo_str})')
            plt.legend()
            #plt.show()

# Print or analyze the best feature combinations for each shape
best_combinations = {shape: max([key for key in accuracy_results if key[0] == shape], key=lambda x: accuracy_results[x]) for shape in shapes}
print("Best feature combinations for each shape:")
for shape, best in best_combinations.items():
    print(f"{shape}: {best[1]} with accuracy {accuracy_results[best]:.2f}")

# %%
#spectral clustring
from sklearn.cluster import SpectralClustering

# Function to match the cluster labels with the session labels
def match_labels(true_labels, predicted_labels):
    D = max(predicted_labels.max(), true_labels.max()) + 1
    cost_matrix = np.zeros((D, D), dtype=int)
    for i in range(len(true_labels)):
        cost_matrix[true_labels[i], predicted_labels[i]] += 1
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    new_predicted_labels = np.zeros_like(predicted_labels)
    for i in range(D):
        new_predicted_labels[predicted_labels == col_ind[i]] = row_ind[i]
    return new_predicted_labels

# Loop through each shape
shapes = ['B', 'M', 'O', 'Star', 'Clover']
features = ['Smoothness_Value', 'Mean_Velocity','Time_Value', 'SSIM_Score', 'Curvature', 'Acceleration', 'Jerk']

# Store the accuracy for each shape and feature combination
accuracy_results = {}

for shape in shapes:
    # Filter the DataFrame for the current shape and create a copy
    shape_df = df[df['Shape'] == shape].copy()
    shape_df = df.copy()

    # Get the true session labels
    true_labels = shape_df['Session'].values

    # Loop through all feature combinations
    for num_features in range(1, len(features) + 1):  # Try 2 to all features
        for feature_combination in combinations(features, num_features):
            # Extract the selected feature combination and create a copy
            X = shape_df[list(feature_combination)].copy()

            # Negate 'Time_Value' if it's in the selected features
            if 'Time_Value' in X.columns:
                X['Time_Value'] = X['Time_Value'] * -1  # Optional negation

            # Perform Spectral Clustering with 2 clusters

            kpca = KernelPCA(kernel='rbf', n_components=2)
            transformed_data = kpca.fit_transform(X)

            spectral_clustering = SpectralClustering(n_clusters=2, assign_labels='discretize', random_state=0)
            spectral_clustering.fit(transformed_data)

            # Add cluster labels to the DataFrame
            shape_df['Cluster'] = spectral_clustering.labels_

            # Predicted labels from Spectral Clustering
            predicted_labels = shape_df['Cluster'].values

            # Match the predicted labels to true labels
            matched_labels = match_labels(true_labels, predicted_labels)

            # Calculate accuracy
            accuracy = accuracy_score(true_labels, matched_labels)
            feature_combo_str = ', '.join(feature_combination)
            print(f"Accuracy for {shape} with features {feature_combo_str}: {accuracy:.2f}")

            # Store the accuracy result
            accuracy_results[(shape, feature_combo_str)] = accuracy

            # Plot the distribution of session numbers in each cluster
            fig, axx = plt.subplots(1, 1)

            # Set bar width for clusters
            bar_width = 0.1
            session_labels = sorted(shape_df['Session'].unique())  # Get sorted session numbers
            indices = np.arange(len(session_labels))  # X positions for the bars

            for cluster in range(2):
                cluster_df = shape_df[shape_df['Cluster'] == cluster]
                session_counts = cluster_df['Session'].value_counts().reindex(session_labels, fill_value=0)

                # Plot the bars for each cluster with an offset
                axx.bar(indices + bar_width * cluster, session_counts.values, bar_width, label=f'Cluster {cluster}')

            # Set x-ticks to be centered between the grouped bars
            axx.set_xticks(indices + bar_width / 2)
            axx.set_xticklabels(session_labels)

            plt.xlabel('Session Number')
            plt.ylabel('Number of Subjects')
            plt.title(f'Distribution of Session Numbers in Each Cluster for {shape} (Features: {feature_combo_str})')
            plt.legend()
            #plt.show()

# Print or analyze the best feature combinations for each shape
best_combinations = {shape: max([key for key in accuracy_results if key[0] == shape], key=lambda x: accuracy_results[x]) for shape in shapes}
print("Best feature combinations for each shape:")
for shape, best in best_combinations.items():
    print(f"{shape}: {best[1]} with accuracy {accuracy_results[best]:.2f}")

# %%

import matplotlib.pyplot as plt
# Loop through each shape
for shape in shapes:
  # Filter the DataFrame for the current shape
  shape_df = df[df['Shape'] == shape]
  shape_df2 = shape_df.copy()

  # Extract the SSIM score and Time value as features
  X = shape_df[['Time_Value', 'SSIM_Score', 'Mean_v']]
  X2 = shape_df[['SSIM_Score', 'Time_Value']]

  #kpca = KernelPCA(kernel='rbf', n_components=2)
  #transformed_data2 = kpca.fit_transform(X2)
  #transformed_data = kpca.fit_transform(X)

  # Perform K-means clustering with 3 clusters
  kmeans = KMeans(n_clusters=2, random_state=0)
  kmeans.fit(X)
  kmeans2 = KMeans(n_clusters=2, random_state=0)
  kmeans2.fit(X2)

  # Add cluster labels to the DataFrame
  shape_df['Cluster'] = kmeans.labels_
  shape_df2['Cluster'] = kmeans2.labels_

  print(shape_df)

  # Plot the distribution of session numbers in each cluster
  fig , axx = plt.subplots(1,2, figsize=(10, 8))
  for cluster in range(2):
    cluster_df = shape_df[shape_df['Cluster'] == cluster]
    print(cluster_df)
    cluster_df2 = shape_df2[shape_df2['Cluster'] == cluster]

    session_counts = cluster_df['Session'].value_counts()
    session_counts2 = cluster_df2['Session'].value_counts()
    print("session_counts", session_counts, "session_counts.index", session_counts.index,"session_counts.values", session_counts.values)
    axx[0].bar(session_counts.index, session_counts.values, label=f'Cluster {cluster}')
    axx[1].bar(session_counts2.index, session_counts2.values, label=f'Cluster {cluster}')

  plt.xlabel('Session Number')
  plt.ylabel('Number of Subjects')
  plt.title(f'Distribution of Session Numbers in Each Cluster for {shape}')
  plt.legend()
  plt.show()


# %%

import matplotlib.pyplot as plt
# Loop through each shape
for shape in shapes:
  # Filter the DataFrame for the current shape
  shape_df = df[df['Shape'] == shape]
  shape_df2 = shape_df.copy()

  # Extract the SSIM score and Time value as features
  X = shape_df[['Time_Value', 'SSIM_Score', 'Mean_v']]

  #kpca = KernelPCA(kernel='rbf', n_components=2)
  #transformed_data2 = kpca.fit_transform(X2)
  #transformed_data = kpca.fit_transform(X)

  # Perform K-means clustering with 3 clusters
  kmeans = KMeans(n_clusters=2, random_state=0)
  kmeans.fit(X)

  # Add cluster labels to the DataFrame
  shape_df['Cluster'] = kmeans.labels_

  print(shape_df)

  # Plot the distribution of session numbers in each cluster
  fig , axx = plt.subplots(1,1)
  for cluster in range(2):
    cluster_df = shape_df[shape_df['Cluster'] == cluster]
    print(cluster_df)

    session_counts = cluster_df['Session'].value_counts()
    print("session_counts", session_counts, "session_counts.index", session_counts.index,"session_counts.values", session_counts.values)
    axx.bar(session_counts.index, session_counts.values, label=f'Cluster {cluster}')

  plt.xlabel('Session Number')
  plt.ylabel('Number of Subjects')
  plt.title(f'Distribution of Session Numbers in Each Cluster for {shape}')
  plt.legend()
  plt.show()


# %%
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame with 'SSIM_Score' and 'Time_Value' columns

for shape in shapes:
  # Filter data for the current shape
  df_shape = df[df['Shape'] == shape]

  # Extract features for clustering
  X = df_shape[['SSIM_Score', 'Time_Value', 'Mean_v', 'Max_v', 'Std_v', 'Smoothness_Value']].values

  # Apply DBSCAN clustering
  dbscan = DBSCAN(eps=0.4, min_samples=3)  # You might need to adjust eps and min_samples
  clusters = dbscan.fit_predict(X)

  # Add cluster labels to the DataFrame
  df_shape['Cluster'] = clusters

  # Plot the results
  plt.figure(figsize=(8, 6))
  for cluster in np.unique(clusters):
    plt.scatter(
        df_shape[df_shape['Cluster'] == cluster]['SSIM_Score'],
        df_shape[df_shape['Cluster'] == cluster]['Time_Value'],
        label=f'Cluster {cluster}'
    )

  plt.xlabel('SSIM Score')
  plt.ylabel('Time Value')
  for i in range(len(df_shape)):
    plt.text(df_shape['SSIM_Score'].iloc[i],
             df_shape['Time_Value'].iloc[i],
             str(df_shape['Session'].iloc[i]),
             fontsize=9, ha='right')
  plt.title(f'DBSCAN Clustering for {shape}')
  plt.legend()
  plt.show()


# %%

import matplotlib.pyplot as plt
import numpy as np
# Assuming 'df' is your DataFrame with 'SSIM_Score' and 'Time_Value' columns

for shape in shapes:
  # Filter data for the current shape
  df_shape = df[df['Shape'] == shape]

  # Extract features for clustering
  X = df_shape[['SSIM_Score', 'Time_Value', 'Mean_v', 'Smoothness_Value']].values

  # Apply DBSCAN clustering


  dbscan = DBSCAN(eps=0.09, min_samples=10)  # You might need to adjust eps and min_samples
  clusters = dbscan.fit_predict(X)

  # Add cluster labels to the DataFrame
  df_shape['Cluster'] = clusters

  # Plot the distribution of session numbers in each cluster
  plt.figure()
  for cluster in np.unique(clusters):
    if cluster != -1:  # Exclude noise points (-1 cluster)
      cluster_df = df_shape[df_shape['Cluster'] == cluster]
      session_counts = cluster_df['Session'].value_counts()
      plt.bar(session_counts.index, session_counts.values, label=f'Cluster {cluster}')
  plt.xlabel('Session Number')
  plt.ylabel('Number of Subjects')
  plt.title(f'Distribution of Session Numbers in Each Cluster for {shape} (DBSCAN)')
  plt.legend()
  plt.show()


# %%

import matplotlib.pyplot as plt
import numpy as np

from sklearn.mixture import GaussianMixture
# Assuming 'df' is your DataFrame with 'SSIM_Score' and 'Time_Value' columns

for shape in shapes:
  # Filter data for the current shape
  df_shape = df[df['Shape'] == shape]

  # Extract features for clustering
  X = df_shape[['SSIM_Score', 'Time_Value','Smoothness_Value', 'Mean_v']].values

  # Apply DBSCAN clustering


# Fit a GMM with 3 components
  gmm = GaussianMixture(n_components=2, covariance_type = 'spherical')
  clusters = gmm.fit_predict(X)

  # Add cluster labels to the DataFrame
  df_shape['Cluster'] = clusters

  # Plot the distribution of session numbers in each cluster
  fig , axx = plt.subplots(1,1, figsize=(10, 8))
  for cluster in range(2):
        cluster_df = df_shape[df_shape['Cluster'] == cluster]
        session_counts = cluster_df['Session'].value_counts().reindex(session_labels, fill_value=0)


        # Plot the bars for each cluster with an offset
        axx.bar(indices + bar_width * cluster, session_counts.values, bar_width, label=f'Cluster {cluster}')



  # Set x-ticks to be centered between the grouped bars
  axx.set_xticks(indices + bar_width / 2)
  axx.set_xticklabels(session_labels)

  plt.xlabel('Session Number')
  plt.ylabel('Number of Subjects')
  plt.title(f'Distribution of Session Numbers in Each Cluster for {shape}')
  plt.legend()
  plt.show()



# %%

import matplotlib.pyplot as plt
import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift, estimate_bandwidth
# Assuming 'df' is your DataFrame with 'SSIM_Score' and 'Time_Value' columns

for shape in shapes:
  # Filter data for the current shape
  df_shape = df[df['Shape'] == shape]

  # Extract features for clustering
  X = df_shape[['SSIM_Score', 'Time_Value']].values

  # Apply DBSCAN clustering


# Fit a GMM with 3 components


  # Estimate the bandwidth based on the data
  bandwidth = estimate_bandwidth(X, quantile=0.1)

  # Perform MeanShift clustering
  mean_shift = MeanShift(bandwidth=bandwidth)
  mean_shift.fit(X)

# Extract cluster centers from MeanShift
  cluster_centers = mean_shift.cluster_centers_

# Now apply k-means on these cluster centers to force exactly 3 clusters
  kmeans = KMeans(n_clusters=3)
  final_labels = kmeans.fit_predict(cluster_centers)

# Assign each data point to the nearest of the 3 clusters found by k-means
# Using the original data and the final cluster centers
  clusters = kmeans.predict(mean_shift.labels_.reshape(-1, 1))

  # Add cluster labels to the DataFrame
  df_shape['Cluster'] = clusters

  # Plot the distribution of session numbers in each cluster
  plt.figure()
  for cluster in np.unique(clusters):
    if cluster != -1:  # Exclude noise points (-1 cluster)
      cluster_df = df_shape[df_shape['Cluster'] == cluster]
      session_counts = cluster_df['Session'].value_counts()
      plt.bar(session_counts.index, session_counts.values, label=f'Cluster {cluster}')
  plt.xlabel('Session Number')
  plt.ylabel('Number of Subjects')
  plt.title(f'Distribution of Session Numbers in Each Cluster for {shape} (DBSCAN)')
  plt.legend()
  plt.show()


# %%
from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans

kpca = KernelPCA(kernel='rbf', n_components=2)
transformed_data = kpca.fit_transform(data)
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(transformed_data)

# %%
import cv2
import os
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt

def compare_images(img1_path, img2_path):
    # Load the two images in grayscale
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        return None  # Skip missing or unreadable images

    # Resize to ensure both images are of the same size
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    # Compute SSIM between the two images
    score, _ = ssim(img1, img2, full=True)
    return score

# Path to your ideal images for each shape (customize as needed)
ideal_images = {
    'B': '/content/drive/MyDrive/project/pics/ideal_B.jpg'
}

# Path to the directory containing your session images
image_dir = '/content/drive/MyDrive/project/pics'

# Dictionary to store SSIM scores
ssim_scores = {shape: {f"session_{m}": [] for m in range(1, 8)} for shape in ideal_images.keys()}

# Loop through all images in the directory
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg"):
        # Parse the filename to get shape, subject, session, and trial info
        parts = filename.split('_')
        shape = parts[0]  # e.g., 'B', 'O', 'M', etc.
        subject = parts[1]  # e.g., 'S1', 'S2', ...
        session = parts[2]  # e.g., 'S1', 'S2', ...
        trial = parts[3]    # e.g., 'T1', 'T2', ...

        # Construct the full path to the image
        image_path = os.path.join(image_dir, filename)

        # Compare with the ideal image for the current shape
        if shape in ideal_images:
            ideal_image_path = ideal_images[shape]
            score = compare_images(ideal_image_path, image_path)
            if score is not None:
                # Store the score in the dictionary
                ssim_scores[shape][session].append(score)

# Calculate average SSIM score for each session
average_ssim = {shape: {} for shape in ssim_scores}
for shape, sessions in ssim_scores.items():
    for session, scores in sessions.items():
        if scores:
            average_ssim[shape][session] = np.mean(scores)
        else:
            average_ssim[shape][session] = None  # No data for this session

# Plotting the average SSIM scores
for shape, sessions in average_ssim.items():
    sessions_list = [f"session_{i}" for i in range(1, 8)]
    avg_scores = [sessions.get(f"session_{i}", None) for i in range(1, 8)]

    plt.figure()
    plt.plot(sessions_list, avg_scores, marker='o')
    plt.title(f"Average SSIM for {shape}")
    plt.xlabel('Sessions')
    plt.ylabel('Average SSIM')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()


