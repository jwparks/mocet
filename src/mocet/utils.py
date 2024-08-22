import re
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate

def fill_nan(A):
    # code adopted from https://stackoverflow.com/a/9815522
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    f = interpolate.interp1d(inds[good], A[good], bounds_error=False, kind='linear')
    B = np.where(np.isfinite(A), A, f(inds))
    return B

def get_avotec_timestamp(time):
    return 3600 * time[0] + 60 * time[1] + time[2] + 0.001 * time[3]

def get_avotec_history(fname, encoding='cp949'):
    global video_duration, record_start_time
    isFirst = True
    historyFile = open(fname, 'r', encoding=encoding)

    token_TTL_high = 'HI'
    token_TTL_low = 'LO'
    token_closing = 'Closing save movie [0]'
    token_start = 'saveMovie[0]:'
    time_between_rec_and_start = 0
    isStarted = False
    line = historyFile.readline()
    time_last_TTL = 0
    while line != '':
        tokens = line.split()
        if len(tokens) > 0:
            if tokens[-1] == token_start:
                record_start_time = tuple([int(token) for token in re.split('[:\.]', tokens[0])])
                isStarted = True
            if token_closing in line:
                record_end_time = tuple([int(token) for token in re.split('[:\.]', tokens[0])])
                video_duration = get_avotec_timestamp(record_end_time) - get_avotec_timestamp(record_start_time)
            if isStarted:
                if (tokens[-1] == token_TTL_high or tokens[-1] == token_TTL_low) and tokens[4] == 'TTL':
                    time = tuple([int(token) for token in re.split('[:\.]', tokens[0])])
                    if isFirst:
                        time_between_rec_and_start = get_avotec_timestamp(time) - get_avotec_timestamp(record_start_time)
                        isFirst = False
                    time_last_TTL = get_avotec_timestamp(time) - get_avotec_timestamp(record_start_time)

        line = historyFile.readline()
    historyFile.close()
    return time_between_rec_and_start, video_duration, time_last_TTL


def clean_avotec_data(log_fname, data_fname, start, duration,
                       end=None, eye_closed_threshold=0.75, encoding='cp949'):
    log_data = pd.read_csv(log_fname)
    log_data = log_data[["center_x", "center_y", "confidence", "diameter_px"]]
    pupil_data_array = log_data.to_numpy()[:, :2]
    pupil_confidence_array = log_data.to_numpy()[:, 2]
    pupil_diameter_array = log_data.to_numpy()[:, 3]

    pupil_dt = []
    with open(data_fname, mode='r', encoding=encoding) as f:
        lines = f.readlines()
        for l, line in enumerate(lines):
            is_recorded = False
            line_split = line.split('\t')
            if line_split[0] == '777':
                pupil_dt.append(np.double(lines[l - 1].split('\t')[2]))
    pupil_dt = np.array(pupil_dt)
    if len(pupil_dt) != len(pupil_data_array):
        if len(pupil_dt) > len(pupil_data_array):
            pupil_dt = pupil_dt[:len(pupil_data_array)]

    if end == None:
        end = start + duration

    for i in range(len(pupil_data_array)):
        if pupil_confidence_array[i] < eye_closed_threshold:
            for k in range(2):
                pupil_data_array[i, :] = np.nan
            pupil_confidence_array[i] = np.nan
    pupil_data_array[:, 0] = fill_nan(pupil_data_array[:, 0])
    pupil_data_array[:, 1] = fill_nan(pupil_data_array[:, 1])

    for i in range(pupil_data_array.shape[0] - 1):
        if np.any(np.isnan(pupil_data_array[i + 1, :])):
            pupil_data_array[i + 1, :] = pupil_data_array[i, :]
    for i in range(pupil_data_array.shape[0] - 1)[::-1]:
        if np.any(np.isnan(pupil_data_array[i, :])):
            pupil_data_array[i, :] = pupil_data_array[i + 1, :]

    pupil_data_array = gaussian_filter1d(pupil_data_array, axis=0, sigma=1)

    pupil_onset = np.cumsum(pupil_dt)
    pupil_effective = np.logical_and(pupil_onset >= start * 1000, pupil_onset < end * 1000)
    try:
        pupil_data_clean = pupil_data_array[pupil_effective, :]
        pupil_timestamps_clean = np.cumsum(pupil_dt[pupil_effective])
        pupil_confidence_clean = pupil_confidence_array[pupil_effective]
        pupil_diameter_clean = pupil_diameter_array[pupil_effective]
    except IndexError:
        try:
            pupil_data_array = pupil_data_array[-(len(pupil_onset)):]
            pupil_data_clean = pupil_data_array[pupil_effective, :]
            pupil_timestamps_clean = np.cumsum(pupil_dt[pupil_effective])

            pupil_confidence_array = pupil_confidence_array[-(len(pupil_onset)):]
            pupil_confidence_clean = pupil_confidence_array[pupil_effective]

            pupil_diameter_array = pupil_diameter_array[-(len(pupil_onset)):]
            pupil_diameter_clean = pupil_diameter_array[pupil_effective]
        except IndexError:
            pupil_effective = pupil_effective[:(len(pupil_data_array))]
            pupil_dt = pupil_dt[:(len(pupil_data_array))]
            pupil_data_clean = pupil_data_array[pupil_effective, :]
            pupil_timestamps_clean = np.cumsum(pupil_dt[pupil_effective])
            pupil_confidence_clean = pupil_confidence_array[pupil_effective]
            pupil_diameter_clean = pupil_diameter_array[pupil_effective]


    return pupil_data_clean, pupil_timestamps_clean, pupil_confidence_clean, pupil_diameter_clean
