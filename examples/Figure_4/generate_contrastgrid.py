import os
import time
import copy
import cv2
import pickle
import numpy as np
from multiprocessing import Pool, get_context

# Video loading function
def load_video_frames(video_path, resize_dims=(1600, 1000)):
    capture = cv2.VideoCapture(video_path)
    frames = []
    while True:
        read, frame = capture.read()
        if not read:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, resize_dims)
        frames.append(frame)
    capture.release()
    return frames


# Function to process each gaze type for a given subject, session, and run
def process_gaze_type(gaze_type, subject, session, task, run, frames, gaze, gaze_dt, fps, framePerTimepoint_toBeUsed,
                      padding_width, padding_height, crop_width, crop_height, imageres_row, imageres_col, npx, addpx):
    stim_acrosstime = np.zeros([imageres_row, imageres_col,
                                len(framePerTimepoint_toBeUsed)])  # Assuming n_TRs == len(framePerTimepoint_toBeUsed)

    frame_count = 0  # Initialize local frame counter
    for tr_count in range(len(framePerTimepoint_toBeUsed)):
        frames_to_be_calculated = []
        for frame_count_inThisTR in range(framePerTimepoint_toBeUsed[tr_count]):
            frame = np.copy(frames[frame_count])
            frame_expand = cv2.copyMakeBorder(frame,
                                              int(padding_height / 2),
                                              int(padding_height / 2),
                                              int(padding_width / 2),
                                              int(padding_width / 2),
                                              cv2.BORDER_CONSTANT,
                                              value=128)

            t = frame_count / fps
            try:
                adj_frame_count = np.min(np.where(gaze_dt >= t * 1000)[0])
                x, y = gaze[adj_frame_count, :]
                x, y = x + padding_width / 2, y + padding_height / 2
                frame_crop = frame_expand[int(y - crop_height / 2):int(y + crop_height / 2),
                             int(x - crop_width / 2):int(x + crop_width / 2)]

                if frame_crop.shape[0] != padding_height or frame_crop.shape[1] != padding_width:
                    frame_expand = cv2.copyMakeBorder(frame, int(padding_height / 2) + addpx,
                                                      int(padding_height / 2) + addpx,
                                                      int(padding_width / 2) + addpx,
                                                      int(padding_width / 2) + addpx,
                                                      cv2.BORDER_CONSTANT, value=128)

                    x, y = gaze[adj_frame_count, :]
                    x, y = x + padding_width / 2 + addpx, y + padding_height / 2 + addpx
                    frame_crop = frame_expand[int(y - crop_height / 2):int(y + crop_height / 2),
                                 int(x - crop_width / 2):int(x + crop_width / 2)]

                frame_crop = cv2.resize(frame_crop, (320, 320))
                frame_crop = (frame_crop.astype('float32') / 255) ** 2
                frames_to_be_calculated.append(frame_crop)
            except:
                pass
            frame_count += 1

        frames_to_be_calculated = np.array(frames_to_be_calculated, dtype=np.float32)
        frames_to_be_calculated -= frames_to_be_calculated.mean(axis=0).reshape(1, 320, 320)

        # Calculate standard deviation in grids for this TR
        temp_acrosstime = np.zeros([imageres_row, imageres_col])
        for rowix in np.arange(imageres_row):
            for colix in np.arange(imageres_col):
                cube = frames_to_be_calculated[:, rowix * npx:(rowix + 1) * npx,
                       colix * npx:(colix + 1) * npx].flatten()
                if np.any(cube > 0):
                    temp_acrosstime[rowix, colix] = np.std(cube)
                else:
                    temp_acrosstime[rowix, colix] = 0.0

        stim_acrosstime[:, :, tr_count] = temp_acrosstime

    # Save the processed data for this gaze type
    np.save(f'contrastgrid/{subject}_{session}_{task}_{run}_contrastgrid_{gaze_type}.npy', stim_acrosstime)
    print(f'Completed processing for {subject} {session} {task} {run} {gaze_type}')


# Main function to process each subject
def process_subject(subject):
    sessions = subject_pool[subject].keys()
    for session in sessions:
        runs, history_loss = subject_pool[subject][session]
        root = f'../../_DATA/{subject}/{session}'
        for r in runs:
            run = f'run-{r}'
            np.random.seed(0)
            key = (subject, session, task, run)
            if key in testable_data.keys():

                n_TRs = 510
                print(f'***************{subject}-{session}-{task}-{run}***************')

                movie_fname = f'{datadir}/{subject}_{session}_{task}_{run}_gameplay.mp4'
                if os.path.exists(movie_fname):
                    frames = load_video_frames(movie_fname)
                    n_frames = len(frames)
                    fps = n_frames / (n_TRs * 1.6)

                    print(f'*************** Video loaded *****************')

                    # Prepare frame per time point calculation
                    framePerTimepoint = finalTR * fps
                    framePerTimepoint_toBeUsed = np.zeros(n_TRs)
                    for tr_count in np.arange(n_TRs):
                        cumulative_frames = np.sum(framePerTimepoint_toBeUsed)
                        groundtruth_frames = framePerTimepoint * (tr_count + 1)
                        now_framePerTimepoint = np.round(groundtruth_frames - cumulative_frames)
                        framePerTimepoint_toBeUsed[tr_count] = now_framePerTimepoint
                    framePerTimepoint_toBeUsed = framePerTimepoint_toBeUsed.astype(int)

                    # Use multiprocessing to process each gaze type in parallel
                    with get_context("spawn").Pool() as pool:
                        results = []
                        for gaze_type in gaze_types:
                            gaze = np.load(
                                f'eyetracking_data/{gaze_type}/{subject}_{session}_{task}_{run}_gaze_coordinate.npy')
                            gaze_dt = np.load(f'eyetracking_data/{subject}_{session}_{task}_{run}_gaze_timestamp.npy')

                            # Start a process for each gaze type
                            results.append(pool.apply_async(process_gaze_type,
                                                            (gaze_type, subject, session, task, run, frames, gaze,
                                                             gaze_dt, fps,
                                                             framePerTimepoint_toBeUsed, padding_width, padding_height,
                                                             crop_width,
                                                             crop_height, imageres_row, imageres_col, npx, addpx)))

                        # Ensure all gaze type processes complete
                        for result in results:
                            result.get()
                else:
                    print('videos not exist')

if __name__ == "__main__":

    finalTR = 1.6

    gaze_types = ['mocet', 'polynomial', 'linear', 'uncorrected']
    subject_pool = {
        'sub-003': {'ses-07R': ([1, 2, 3, 4, 5], False),
                    'ses-13R': ([1, 2, 4, 5, 6], False)},
        'sub-004': {'ses-07R': ([1, 2, 3, 4, 5, 6], False),
                    'ses-13': ([1, 2, 3, 4, 5, 6], False)},
        'sub-005': {'ses-07': ([1, 2, 3, 4, 5, 6], True)},
        'sub-006': {'ses-07R': ([1, 2, 3, 4, 5, 6], False),
                    'ses-13': ([1, 2, 3, 4, 5, 6], False)},
        'sub-008': {'ses-07R': ([2, 3, 4, 5, 6], False),
                    'ses-13': ([1, 2, 3, 4, 5, 6], False)},
        'sub-009': {'ses-07': ([1, 2, 3, 4, 5, 6], False),
                    'ses-13': ([1, 2, 3, 5, 6], False)},
        'sub-010': {'ses-07': ([1, 2, 3, 4, 5], False),
                    'ses-13': ([1, 2, 3, 4, 5, 6], False)},
        'sub-011': {'ses-07': ([1, 2, 3, 4, 5, 6], False),
                    'ses-13': ([1, 2, 3, 4, 5, 6], False)},
        'sub-012': {'ses-07': ([1, 2, 4, 5, 6], False)},
        'sub-013': {'ses-07': ([1, 2, 3, 4], False)},
        'sub-014': {'ses-07': ([2, 3, 4, 5, 6], False)},
        'sub-015': {'ses-07': ([1, 2, 3, 4, 5, 6], False),
                    'ses-13': ([1, 2, 3, 4, 5, 6], False)},
        'sub-016': {'ses-07': ([1, 2, 3, 4, 5, 6], False),
                    'ses-13': ([1, 2, 3, 4, 5, 6], False)},
        'sub-017': {'ses-07': ([1, 2, 3, 4, 5, 6], False),
                    'ses-13': ([1, 2, 3, 4, 5], False)},
        'sub-018': {'ses-07': ([1, 2, 3, 4, 5, 6], False),
                    'ses-13': ([1, 2, 3, 4, 5, 6], False)},
        'sub-020': {'ses-07': ([1, 2, 3, 4, 5, 6], False),
                    'ses-13': ([1, 2, 3, 4, 5, 6], False)},
        'sub-021': {'ses-07': ([1, 2, 4, 5, 6], False),
                    'ses-13': ([1, 2, 4, 5, 6], False)},
        'sub-JJY': {'ses-07': ([1, 2, 3, 4, 5, 6], False)},
        'sub-KMY': {'ses-07': ([1, 2, 3, 4, 5, 6], False)},
        'sub-PJW': {'ses-07': ([1, 2, 3, 4, 6], True)},
        'sub-PBJ': {'ses-07': ([1, 2, 3, 4, 5], False)}
    }

    task = 'task-mcHERDING'
    method = 'linear'
    testable_data = pickle.load(open('../testable_data_list.pkl', 'rb'))

    subjects = []
    for key in list(testable_data.keys()):
        subjects.append(key[0])
    subjects = list(set(subjects))
    subjects.sort()
    # Process each subject's data

    datadir = '/Volumes/HAZE/gameplay_resize.nosync'

    padding_width, padding_height = 1600, 2200
    crop_width, crop_height = 1600, 1600
    imageres_row = 64  # 400 # the number of rows of total grids
    imageres_col = 64  # 640 # the number of colums of total grids
    npx = 5  # 5 # pixel per grid width (or height)
    addpx = 1600

    for subject in subjects:
        process_subject(subject)