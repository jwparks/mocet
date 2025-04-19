# Motion-corrected eyetracking (MoCET)
This repository contains the code for the paper  "Park, J., Jeon, J. Y., Kim, R., Kay, K. & Shim, W. M. Motion-corrected eye tracking (MoCET) improves gaze accuracy during visual fMRI experiments. bioRxiv (2025) doi:10.1101/2025.03.13.642919" 

The **MoCET (Motion-Corrected Eye-Tracking)** Python package provides tools for compensating head movement-induced errors in eye-tracking data collected during fMRI experiments. This package integrates motion correction techniques with advanced eye-tracking algorithms to enhance gaze accuracy, particularly in dynamic environments where head movement is common.

### Key Features:
- **Head Motion Compensation**: Implements a robust algorithm that leverages head motion parameters from fMRI data preprocessing to correct gaze position errors caused by head movements.
- **Simulation Support**: Includes tools for simulating head motion and its impact on eye-tracking data, enabling validation and testing of the correction algorithms.

### Installation:
MoCET can be installed via pip:
```python
pip install mocet
```

### MoCET for Arrington/Avotec systems
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
  - [In the data repository](https://zenodo.org/records/14892082), we provide all pupil coordinate log files, data files, history files, and event files necessary for replicating the study.
  - Due to storage limitations, the raw eye video files (totaling over 1.8TB) are not included. Please contact us if you require access to the raw video data.
  
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

#### Applying MoCET (Motion-Correction Eye Tracking)
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

start, _, _ = mocet.utils.get_avotec_history(history_fname)
pupil_data, pupil_timestamps, pupil_confidence, _ = mocet.utils.clean_avotec_data(log_fname,
                                                                             data_fname,
                                                                             start=start,
                                                                             duration=task_duration)

# Apply the motion correction using confounds data from fMRIprep
confounds_fname = f'{subject}_{session}_{task}_{run}_desc-confounds_timeseries.tsv'
pupil_data = mocet.apply_mocet(pupil_data, confounds_fname, large_motion_params=False, polynomial_order=3)
```

For more detailed information on how to use MoCET, please refer to the `examples/Applying_MoCET.ipynb` notebook.

### MoCET for EyeLink system
The data acquisition and preprocessing steps for SR Research’s EyeLink system differ significantly from the Avotec/Arrington system. Unlike the latter, which primarily records pupil coordinates, EyeLink internally determines gaze coordinates during an experiment. This is because it detects the pupil and applies an internal calibration model to convert pupil position into gaze location on the screen.

#### Considerations for Using MoCET with EyeLink Data
- **Ensure raw pupil coordinates are recorded**: The EyeLink system allows for recording raw pupil coordinates before they are transformed into gaze coordinates. To enable this, users should set the `File Sample Contents` option to `Raw Eye Position` in the EyeLink configuration.
- **Do not apply EyeLink’s Drift Correction**: We strongly recommend against using internal Drift Correction function in the EyeLink system, as it modifies the gaze data in ways that may interfere with MoCET’s correction approach.

We don’t currently collect EyeLink data ourselves, but if you’re interested in applying MoCET to your EyeLink dataset, we’d love your help! If you can share an example dataset with us, specifically:
- Raw pupil coordinates from EyeLink (set the `File Sample Contents` option to `Raw Eye Position` in the EyeLink configuration)
- Head motion parameters from your fMRI preprocessing (e.g., from fMRIprep, FSL, or SPM)

This would allow us to develop a tailored MoCET pipeline for EyeLink users, making it easier to apply our method to your data. Your contribution would help us ensure accurate head motion correction for EyeLink-based eye tracking in fMRI. If you’re interested or have any questions, feel free to reach out! [jiwoongpark@skku.edu](mailto:jiwoongpark@skku.edu)

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
displacement = mocet.simulation.generate(motion_params, camera_parameters, 'path/to/output/', index=0, render=True)
```
For more detailed information on how to use the simulation tools, please refer to the `analysis/Figure_2/generate_model_simulation.ipynb` notebook.

### Key dependencies:
- Python 3.11.4
- NumPy (`numpy==1.25.2`)
- SciPy (`scipy==1.11.2`)
- pandas (`pandas==2.2.3`)
- scikit-learn (`scikit-learn==1.3.0`)
- statsmodels (`statsmodels==0.14.4`)
- gstreamer (`gstreamer==1.14.1`)
- matplotlib (`matplotlib==3.7.1`)
- seaborn (`seaborn==0.12.2`)

### References
- Park, J., Jeon, J. Y., Kim, R., Kay, K. & Shim, W. M. Motion-corrected eye tracking (MoCET) improves gaze accuracy during visual fMRI experiments. bioRxiv 2025.03.13.642919 doi:10.1101/2025.03.13.642919.
- Santini, T., Fuhl, W. & Kasneci, E. PuReST: robust pupil tracking for real-time pervasive eye tracking. Proc. 2018 ACM Symp. Eye Track. Res. Appl. 1–5 (2018) doi:10.1145/3204493.3204578.
- OpenPuPilExt: https://github.com/openPupil/Open-PupilEXT
- eyerec-python: https://github.com/tcsantini/eyerec-python
  
