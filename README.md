# Motion-corrected eyetracking (MoCET)

The MoCET (Motion-Corrected Eye-Tracking) Python package provides tools for compensating head movement-induced errors in eye-tracking data collected during fMRI experiments. This package integrates motion correction techniques with advanced eye-tracking algorithms to enhance gaze accuracy, particularly in dynamic environments where head movement is common.

### Key Features:
- **Head Motion Compensation**: Implements a robust algorithm that leverages head motion parameters from fMRI data preprocessing to correct gaze position errors caused by head movements.
- **Simulation Support**: Includes tools for simulating head motion and its impact on eye-tracking data, enabling validation and testing of the correction algorithms.

### Installation:
MoCET can be installed via pip:
```python
pip install mocet
```

### For Avotec system
```python
import mocet

subject = 'sub-001'
session = 'ses-01'
task = 'task-example'
run = 'run-1'

# Load your eye-tracking data and cleaning
log_fname = f'{subject}_{session}_{task}_{run}_recording-eyetracking_physio_log.csv'
data_fname = f'{subject}_{session}_{task}_{run}_recording-eyetracking_physio_dat.txt' 
history_fname = f'{subject}_{session}_{task}_{run}_recording-eyetracking_physio_his.txt'
start, _, _ = mocet.get_avotec_history(history_fname)
pupil_data, pupil_timestamps, pupil_confidence, _ = mocet.clean_avotec_data(log_fname,
                                                                             data_fname,
                                                                             start=start,
                                                                             duration=task_duration)

# Apply the motion correction using confounds data from fMRIprep
confounds_fname = f'{root}/{subject}_{session}_{task}_{run}_desc-confounds_timeseries.tsv'
pupil_data = mocet.apply_mocet(pupil_data, confounds_fname,
                               large_motion_params=False,
                               polynomial_order=0)

```

### For EyeLink system
```python
# TODO
```
