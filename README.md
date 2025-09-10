# Motion-corrected eyetracking (MoCET)
This repository contains the code for the paper  "Park, J., Jeon, J. Y., Kim, R., Kay, K. & Shim, W. M. Motion-corrected eye tracking (MoCET) improves gaze accuracy during visual fMRI experiments. bioRxiv (2025) doi:10.1101/2025.03.13.642919" 

The **MoCET (Motion-Corrected Eye-Tracking)** Python package provides tools for compensating head movement-induced errors in eye-tracking data collected during fMRI experiments. This package integrates motion correction techniques with advanced eye-tracking algorithms to enhance gaze accuracy, particularly in dynamic environments where head movement is common.

### Key Features:
- **Head Motion Compensation**: Implements a robust algorithm that leverages head motion parameters from fMRI data preprocessing to correct gaze position errors caused by head movements.
- **Across-Run Calibration Transfer**: Supports an across-run variant of MoCET, enabling calibration transfer within the same scan session without requiring repeated calibrations per run.
- **Simulation Support**: Includes tools for simulating head motion and its impact on eye-tracking data, enabling validation and testing of the correction algorithms.

### Installation:
MoCET can be installed via pip:
```bash
pip install mocet
```

### Usage:
#### Eye Tracking Data Acquisition
- MoCET was developed using **camera-based eye tracking data recorded as raw pupil coordinates**. In this study, we used Avotec’s MR-compatible eye tracking hardware and Arrington Research’s ViewPoint recording software.
- ViewPoint recording software generates four output files:
  - **Raw eye video file (*.avi)**: Contains the recorded eye tracking video.
  - **Data file (*.txt)**: Stores time interval information between video frames.
  - **History file (*.txt)**: Logs TTL signals from the MRI scanner.
  - **Event file (*.txt)**: Logs detected eye movements such as saccades and fixations.
- Instead of using the event file, we extract eye movement through post-processing of the video file, as it provides more reliable pupil detection.

#### Eye Tracking Data Preprocessing
- We recommand to use the [PuReST algorithm (Santini et al., 2018)](https://doi.org/10.1145/3204493.3204578) from the open-source [pupilEXT](https://github.com/openPupil/Open-PupilEXT) software for GUI-basesd accurate pupil detection.
- However, for seemless integration with python and MoCET, we recommend OpenCV and [eyerec-python](https://github.com/tcsantini/eyerec-python) for pupil extraction via the command line.
- For example eye video data to test pupil extraction, please refer to this [link](https://drive.google.com/file/d/1Q8PsecMtoM5hY7cPPaA7pmVjM_eyMRnr/view?usp=sharing).
  - Note that the full raw eye video file exceeds 16GB for a ~13-minute run, so we provide a trimmed version (1GB, 60s) for demonstration purposes.
  - [In the data repository](https://zenodo.org/records/17089244), we provide all files necessary for replicating the study.
  - Due to storage limitations, the raw eye video files (totaling over 1.8TB) are not included. Please contact us if you require access to the raw video data.

Example code for installing eyerec-python (Thanks Royoung!), but you may need some troubleshooting to install the C++ library. For more details, please refer to the [eyerec-python](https://github.com/tcsantini/eyerec-python) documentation.
```bash
git clone https://github.com/tcsantini/eyerec-python.git
cd eyerec-python/lib/cpp

g++ -O2 -Wall -c -fPIC -I${eyerec_dir}/lib/cpp/src -I/usr/include/opencv4 \
		-I./src -I./src/common -I./include \
    src/common/ocv_utils.cpp \
    src/pupil/detection/PupilDetectionMethod.cpp \
    src/pupil/detection/PuRe/PuRe.cpp \
    src/pupil/tracking/PupilTrackingMethod.cpp \
    src/pupil/tracking/PuReST/PuReST.cpp
  
ar rcs libeyerec_cpp.a *.o
cd ../..
cython --cplus -3 eyerec/_eyerec.pyx

pip install -e .
```

Example of pupil extraction using OpenCV and eyerec-python:
```python
import cv2
import eyerec
import numpy as np
import pandas as pd

subject = 'sub-001'
session = 'ses-01'
task = 'task-example'
run = 'run-1'

filename = f'{subject}_{session}_{task}_{run}_recording-eyetracking_physio_mv.avi'
capture = cv2.VideoCapture(filename)
read, frame = capture.read()

tracker = eyerec.PupilTracker(name='purest')

df = pd.DataFrame(columns=['diameter_px','width_px','height_px','axisRatio',
                           'center_x','center_y', 'angle_deg', 'confidence'])
count = 0
while True:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pupil = tracker.detect(capture.get(cv2.CAP_PROP_POS_MSEC), frame)
    
    df.loc[count] = {'diameter_px':np.max([pupil['size']]),
                     'width_px':pupil['size'][0],
                     'height_px':pupil['size'][1],
                     'axisRatio':np.min([pupil['size']])/np.max([pupil['size']]),
                     'center_x':pupil['center'][0],
                     'center_y':pupil['center'][1],
                     'angle_deg': pupil['angle'],
                     'confidence':pupil['confidence']}
    
    read, frame = capture.read()
    if not read:
        break
    count += 1
```
Finally save the pupil data to a csv file:
```python
target_filename = f'{subject}_{session}_{task}_{run}_recording-eyetracking_physio_log.csv'
df.to_csv(target_filename, index=False)
```
For more detailed information on how to use eyerec-python, please refer to the `examples/pupil_tracking,ipynb` notebook. 

#### Motion Parameters Extraction from fMRI data
- MoCET requires head motion parameters from fMRI data preprocessing to correct gaze position errors caused by head movements.
- We recommend using the [fMRIprep](https://fmriprep.org/en/stable/) software for fMRI data preprocessing, as it provides head motion parameters in the confounds file.
- The confounds file contains the following 6-DoF motion parameters:
  - `trans_x`, `trans_y`, `trans_z`: Translation in the x, y, and z directions.
  - `rot_x`, `rot_y`, `rot_z`: Rotation around the x, y, and z axes.
- If you're using FSL for fMRI preprocessing, you can extract motion parameters using the `mcflirt` tool. The output file will contain similar motion parameters that can be used with MoCET with `motion_source='mcflirt'` option.

#### Applying MoCET for Avotec/Arrington system
- MoCET can be applied to the eye tracking data using the following steps:
  1. Load the eye tracking data and clean it.
  2. Extract the start time of the eye tracking data from the history file.
  3. Apply the motion correction using the head motion parameters from the confounds file.

```python
import mocet

# Load your eye tracking data and cleaning
log_fname = f'{subject}_{session}_{task}_{run}_recording-eyetracking_physio_log.csv'
data_fname = f'{subject}_{session}_{task}_{run}_recording-eyetracking_physio_dat.txt' 
history_fname = f'{subject}_{session}_{task}_{run}_recording-eyetracking_physio_his.txt'
task_duration = 816  # in seconds

start, _, _ = mocet.utils.get_viewpoint_history(history_fname)
pupil_data, pupil_timestamps, pupil_confidence, _ = mocet.utils.clean_viewpoint_data(log_fname,
                                                                             data_fname,
                                                                             start=start,
                                                                             duration=task_duration)

# Apply the motion correction using confounds data from fMRIprep
confounds_fname = f'{subject}_{session}_{task}_{run}_desc-confounds_timeseries.tsv'
pupil_data = mocet.apply_mocet(pupil_data, 
                               motion_params_fname=confounds_fname, 
                               pupil_confidence=pupil_confidence, # optional but recommended
                               motion_source='fmriprep', # or 'mcflirt' for FSL output files (.par)
                               polynomial_order=3)
```
For more detailed information on how to use MoCET, please refer to the `analysis/scripts/generate_corrected_eyetracking.ipynb` notebooks.

#### Nonlinear MoCET variants
We also provide two nonlinear variants of MoCET (MoCET-Large and MoCET-Interaction) for detrending the nonlinear relationship between head motion and gaze position errors.
- **MoCET-Large**, which includes 24 regressors, combining the original 6 DoF motion parameters with their temporal derivatives, squared terms, and squared derivatives, along with polynomial terms up to cubic order.
- **MoCET-Interaction**, which adds all pairwise second-order interaction terms among the 6 motion parameters, in addition to the original set and the same polynomial terms.

```python
# For MoCET-Large
pupil_data = mocet.apply_mocet(pupil_data, 
                               motion_params_fname=confounds_fname, 
                               pupil_confidence=pupil_confidence,
                               large_motion_params=True, 
                               motion_source='fmriprep',
                               polynomial_order=3)

# For MoCET-Interaction
pupil_data = mocet.apply_mocet(pupil_data, 
                               motion_params_fname=confounds_fname, 
                               pupil_confidence=pupil_confidence,
                               interactions=True, 
                               motion_source='fmriprep',
                               polynomial_order=3)
```
For more detailed information on how to use nonlinear MoCET, please refer to the `analysis/scripts/nonlinear_mocet/motion_types.ipynb` notebook.

#### Across-run MoCET
We support an across-run variant of MoCET, which enables calibration transfer within the same scan session. This method demonstrated high correction accuracy without requiring repeated calibrations per run.
To use across-run MoCET, you need to apply motion correction across multiple runs within the same session. Here's an example of applying across-run motion correctio, use a first run as a reference for subsequent runs:

Important note: This feature is not yet functional in the current MoCET package version (v0.1.0). Therefore, we provide the code snippet below for users to implement it manually.
```bash
# merge multiple fMRI data using fslmerge
fslmerge -t {subject}_{session}_{task}_allruns_bold.nii.gz \
            {subject}_{session}_{task}_run-1_bold.nii.gz \
            {subject}_{session}_{task}_run-2_bold.nii.gz \
            {subject}_{session}_{task}_run-3_bold.nii.gz 
            
# applying motion correction using mcflirt (replace -refvol with your desired reference volume, here 0 indicates the first volume of the first run)
mcflirt -in {subject}_{session}_{task}_allruns_bold.nii.gz -plots -refvol 0
```

After running the above commands, you will get a motion parameter file named `{subject}_{session}_task-{task}_allruns_bold_mcf.par`. You can then use this file for across-run MoCET as follows:

```python
def get_mcflirt_motion_params(motion_params_fname, use_mm_deg = False):
    motion_params = np.genfromtxt(motion_params_fname)
    rotations = motion_params[:, :3] # in radians
    translations = motion_params[:, 3:6] # in mm
    if use_mm_deg:
        rotations = np.rad2deg(rotations)
    motion_params = np.hstack((translations, rotations))
    return motion_params

def polynomial_detrending(pupil_data, polynomial_order):
    X = make_poly_regressors(len(pupil_data), order=polynomial_order)
    dedrift_regressor = np.zeros((len(pupil_data), 2))
    for i in range(2):
        reg = LinearRegression(fit_intercept=False).fit(X, pupil_data[:, i])
        dedrift_regressor[:, i] = reg.predict(X)
    pupil_data = pupil_data[:, :2] - dedrift_regressor
    return pupil_data

merged_motion_params_fname = f'{subject}_{session}_{task}_allruns_bold_mcf.par'
merged_motion_params = get_mcflirt_motion_params(merged_motion_params_fname)

# split the merged motion parameters into individual runs
runs = ['run-1', 'run-2', 'run-3']
num_volumes_per_run = [510, 510, 510] # replace with the actual number of volumes in each run
motion_params = np.split(merged_motion_params, np.cumsum(num_volumes_per_run)[:-1])

# Load Run 1 eye tracking data and fit the calibration model
ref_run_idx = 0 # index of the reference run which has calibration data
run = runs[ref_run_idx]
log_fname = f'{subject}_{session}_{task}_{run}_recording-eyetracking_physio_log.csv'
data_fname = f'{subject}_{session}_{task}_{run}_recording-eyetracking_physio_dat.txt'
history_fname = f'{subject}_{session}_{task}_{run}_recording-eyetracking_physio_his.txt'
start, _, _ = mocet.utils.get_viewpoint_history(history_fname)
task_duration = 816  # in seconds

pupil_data, pupil_timestamps, pupil_confidence, pupil_diameter = mocet.utils.clean_viewpoint_data(
    log_fname,
    data_fname,
    start=start,
    duration=task_duration)

# Resample motion parameters to match the length of pupil data
motion_params_resample = np.zeros((len(pupil_data), motion_params[ref_run_idx].shape[1]))
x = np.arange(0, len(motion_params[ref_run_idx]))
for i in range(motion_params[ref_run_idx].shape[1]):
    y = motion_params[ref_run_idx][:, i]
    f = interpolate.interp1d(x, y)
    xnew = np.linspace(0, len(motion_params[ref_run_idx]) - 1, len(pupil_data))
    motion_params_resample[:, i] = f(xnew)

# Fit the calibration model using the reference run
X = np.hstack((run_motion_params[best_run], make_poly_regressors(len(motion_params), order=3)))

coefs_ = []
models = []
dedrift_regressor = np.zeros((len(pupil_data), 2))
for i in range(2):
    reg = LinearRegression(fit_intercept=False).fit(X, pupil_data[:, i])
    coefs_.append(reg.coef_)
    models.append(reg)
    dedrift_regressor[:, i] = reg.predict(X)
pupils_corrected = pupil_data[:, :2] - dedrift_regressor
pupils_corrected_mean = np.mean(pupils_corrected, axis=0)

offset = calibration_onsets[0]
calibration_pupils = []
for i in np.arange(calibration_points[0]):
    start = (offset+i)*interval + calibration_offset_start
    end = (offset+i+1)*interval + calibration_offset_end
    log_effective = np.logical_and(pupil_timestamps >= start*1000, pupil_timestamps < end*1000)
    calibration_pupils.append([np.nanmean(pupils_corrected[log_effective,0]),
                              np.nanmean(pupils_corrected[log_effective,1])])
calibration_pupils = np.array(calibration_pupils)

calibrator = mocet.EyetrackingCalibration(calibration_coordinates = calibration_coordinates,
                                          calibration_order = calibration_order,
                                          repeat=True)
calibrator.fit(calibration_pupils[:, 0], calibration_pupils[:, 1])

# Apply the across-run MoCET to other runs using the fitted model from the reference run
for run_idx, run in enumerate(runs):
    if run_idx == ref_run_idx:
        continue  # Skip the reference run as it has already been processed
    log_fname = f'{subject}_{session}_{task}_{run}_recording-eyetracking_physio_log.csv'
    data_fname = f'{subject}_{session}_{task}_{run}_recording-eyetracking_physio_dat.txt'
    history_fname = f'{subject}_{session}_{task}_{run}_recording-eyetracking_physio_his.txt'
    start, _, _ = mocet.utils.get_viewpoint_history(history_fname)
    task_duration = 816  # in seconds
    
    pupil_data, pupil_timestamps, pupil_confidence, pupil_diameter = mocet.utils.clean_viewpoint_data(
        log_fname,
        data_fname,
        start=start,
        duration=task_duration)
    
    motion_params_resample = np.zeros((len(pupil_data), motion_params[ref_run_idx].shape[1]))
    x = np.arange(0, len(motion_params[ref_run_idx]))
    for i in range(motion_params[ref_run_idx].shape[1]):
        y = motion_params[ref_run_idx][:, i]
        f = interpolate.interp1d(x, y)
        xnew = np.linspace(0, len(motion_params[ref_run_idx]) - 1, len(pupil_data))
        motion_params_resample[:, i] = f(xnew)
    
    X = motion_params_resample
    dedrift_regressor = np.zeros((len(pupil_data), 2))
    for i in range(2):
        dedrift_regressor[:, i] = X @ coefs_[i][:6] # use the coefficients fitted using the reference run
    pupils_corrected = pupil_data[:, :2] - dedrift_regressor
    pupils_corrected = polynomial_detrending(pupils_corrected, polynomial_order=3) # apply polynomial detrending
    gaze_coordinates = calibrator.transform(pupils_corrected) # use the calibrator fitted using the reference run
```
For more detailed information on how to use across-run MoCET, please refer to the `analysis/scripts/cross_alignment/cross_alignment_analysis.ipynb` notebook.

#### Applying MoCET for EyeLink system
The data acquisition and preprocessing steps for SR Research’s EyeLink system differ significantly from the Avotec/Arrington system. Unlike the latter, which primarily records pupil coordinates, EyeLink internally determines gaze coordinates during an experiment. This is because it detects the pupil and applies an internal calibration model to convert pupil position into gaze location on the screen.
- **Ensure raw pupil coordinates are recorded**: The EyeLink system allows for recording raw pupil coordinates before they are transformed into gaze coordinates. To enable this, users should set the `File Sample Contents` option to `Raw Eye Position` in the EyeLink configuration.
- **Do not apply EyeLink’s Drift Correction**: We strongly recommend against using internal Drift Correction function in the EyeLink system, as it modifies the gaze data in ways that may interfere with MoCET’s correction approach.

```python
import mocet
from eyelinkio import read_edf
from scipy.signal import decimate

def forward_fill(arr):
    arr = arr.copy()
    nanmask = np.isnan(arr[:, 0])
    for col in range(arr.shape[1]):
        mask = np.isnan(arr[:, col])
        valid = ~mask
        if not np.any(valid):
            continue  # skip if all NaN
        arr[mask, col] = np.interp(np.flatnonzero(mask), np.flatnonzero(valid), arr[valid, col])
    return arr, nanmask.astype(np.float64)

def remove_spike(arr, duration = 5, fs=100, z = 2.0):
    window = duration*fs
    for col in range(arr.shape[1]):
        for i in range(len(arr[:,col]) - 1):
            start_idx = max(i - window // 2, 0)
            end_idx = min(i + window // 2, len(arr[:,col]))
            local_mean = np.mean(arr[start_idx:end_idx, col])
            local_std = np.std(arr[start_idx:end_idx, col])
            if  np.logical_or(arr[i + 1, col] < (local_mean - z * local_std),
                              arr[i + 1, col] > (local_mean + z * local_std)):
                arr[i + 1, col] = arr[i, col]
    return arr

fs_orig = 1000  # original sampling rate
fs_new = 100    # desired sampling rate
factor = int(fs_orig / fs_new)

# Load your EyeLink data
data_fname = f'{subject}_{session}_{task}_{run}_recording-eyetracking.edf'
eyelink_data = read_edf(data_fname)

# Preprocess the raw pupil coordinates
signal, nanmask = forward_fill(eyelink_data['samples'][:2,:].T)
signal = decimate(signal, factor, ftype='iir', zero_phase=True, axis=0)
nanmask = decimate(nanmask, factor, ftype='iir', zero_phase=True, axis=0) >= 0.75
signal = remove_spike(signal)

# Extract data segment corresponding to the task duration
start_idx = 0 # Modify this if there's a specific start time
duration = 816  # in seconds
pupil_data = np.copy(signal[start_idx:int(start_idx+task_duration*fs),:])
pupil_timestamps = np.arange(0,task_duration,1/fs)*1000
pupil_isnan = np.copy(nanmask[start_idx:int(start_idx+task_duration*fs)])

# Apply the MoCET using confounds data from fMRIprep
confounds_fname = f'{subject}_{session}_{task}_{run}_desc-confounds_timeseries.tsv'
pupil_data = mocet.apply_mocet(pupil_data, confounds_fname, polynomial_order=3)
```

### Computational Simulation of Eye Movements
- MoCET provides tools for simulating head motion and its impact on eye-tracking data, enabling validation and testing of the correction algorithms.
- The simulation process involves generating synthetic head motion parameters and applying them to the eye-tracking data to create motion-induced gaze errors.

```python
import mocet

motion_param_labels = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']

fmriprep_confounds = pd.read_csv(confounds_fname, delimiter='\t')
motion_params = fmriprep_confounds[motion_param_labels]
motion_params = np.nan_to_num(motion_params)
motion_params = motion_params - motion_params[0, :]
basis_params = [0, 0, 0.1, 0] # yaw, pitch, distance, roll
displacement, pupil_coordinates = mocet.simulation.generate(motion_params, basis_params,
                                     render=True, render_resolution = (128, 96), detect_pupil=True)
```
For more detailed information on how to use the simulation tools, please refer to the `analysis/scripts/model_simulation/generate_simulation_data.ipynb` notebook.


## Key dependencies:
- Python 3.11.4
- NumPy (`numpy==1.25.2`)
- SciPy (`scipy==1.11.2`)
- pandas (`pandas==2.2.3`)
- scikit-learn (`scikit-learn==1.3.0`)
- statsmodels (`statsmodels==0.14.4`)
- gstreamer (`gstreamer==1.14.1`)
- matplotlib (`matplotlib==3.7.1`)
- seaborn (`seaborn==0.12.2`)

## References
- Park, J., Jeon, J. Y., Kim, R., Kay, K. & Shim, W. M. Motion-corrected eye tracking (MoCET) improves gaze accuracy during visual fMRI experiments. bioRxiv 2025.03.13.642919 doi:10.1101/2025.03.13.642919.
- Santini, T., Fuhl, W. & Kasneci, E. PuReST: robust pupil tracking for real-time pervasive eye tracking. Proc. 2018 ACM Symp. Eye Track. Res. Appl. 1–5 (2018) doi:10.1145/3204493.3204578.
- OpenPuPilExt: https://github.com/openPupil/Open-PupilEXT
- eyerec-python: https://github.com/tcsantini/eyerec-python
  
