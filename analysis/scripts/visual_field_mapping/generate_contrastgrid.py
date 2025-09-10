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

def load_movie(key):
    subject, session, task, run = key
    movie_dir = '/DATA/Minecraft_old/_DATA/gameplay_video'
    movie_fname = f'{movie_dir}/{subject}_{session}_{task}_{run}_gameplay.mp4'
    if os.path.exists(movie_fname):
        print(f'Loading movie: {movie_fname}')

        frames = load_video_frames(movie_fname)
        n_frames = len(frames)
        fps = n_frames / (n_TRs * 1.6)
        return frames, fps, key
    else:
        return None, None, key

# Function to process each gaze type for a given subject, session, and run
def calculate_contrastgrid(gaze_type, subject, session, task, run):
    output_dir = f'../../data/contrastgrid/{subject}/{session}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(f'{output_dir}/{subject}_{session}_{task}_{run}_contrastgrid_{gaze_type}.npy'):
        pass
    else:
        gaze_root = f'../../data/corrected_eyetracking/{gaze_type}/{subject}/{session}'
        gaze = np.load(f'{gaze_root}/{subject}_{session}_{task}_{run}_gaze_coordinate.npy')
        gaze_dt = np.load(f'{gaze_root}/{subject}_{session}_{task}_{run}_gaze_timestamp.npy')
        frames, fps = movie_data[(gaze_type, subject, session, task, run)]

        framePerTimepoint = finalTR * fps
        framePerTimepoint_toBeUsed = np.zeros(n_TRs)
        for tr_count in np.arange(n_TRs):
            cumulative_frames = np.sum(framePerTimepoint_toBeUsed)
            groundtruth_frames = framePerTimepoint * (tr_count + 1)
            now_framePerTimepoint = np.round(groundtruth_frames - cumulative_frames)
            framePerTimepoint_toBeUsed[tr_count] = now_framePerTimepoint
        framePerTimepoint_toBeUsed = framePerTimepoint_toBeUsed.astype(int)

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
                                                  value=128) # frame becomes 640 x 640

                t = frame_count / fps

                try:
                    adj_frame_count = np.min(np.where(gaze_dt >= t * 1000)[0])
                    x, y = gaze[adj_frame_count, :] # gaze location in (1600, 1000) coordinates
                    x, y = x + padding_width / 2, y + padding_height / 2
                    frame_crop = frame_expand[int(y - crop_height / 2):int(y + crop_height / 2),
                                 int(x - crop_width / 2):int(x + crop_width / 2)]

                    if frame_crop.shape[0] != crop_height or frame_crop.shape[1] != crop_width:
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
                    frame_crop = (frame_crop.astype(np.float16) / 255) ** 2
                    frames_to_be_calculated.append(frame_crop)
                except:
                    pass
                frame_count += 1

            temp_acrosstime = np.zeros([imageres_row, imageres_col])
            if len(frames_to_be_calculated) == 0:
                print(f'No frames to calculate for {subject} {session} {task} {run} {gaze_type} TR {tr_count}')
            else:
                frames_to_be_calculated = np.array(frames_to_be_calculated, dtype=np.float16)
                #frames_to_be_calculated -= frames_to_be_calculated.mean(axis=0).reshape(1, 640, 640)
                for rowix in np.arange(imageres_row):
                    for colix in np.arange(imageres_col):
                        cube = frames_to_be_calculated[:, rowix * npx:(rowix + 1) * npx, colix * npx:(colix + 1) * npx].flatten()
                        if not np.allclose(cube, cube[0]):
                            cube_mean = np.mean(cube)
                            if cube_mean != 0:
                                temp_acrosstime[rowix, colix] = np.std(cube) / cube_mean
                            else:
                                temp_acrosstime[rowix, colix] = 0.0
                        else:
                            temp_acrosstime[rowix, colix] = 0.0

            stim_acrosstime[:, :, tr_count] = temp_acrosstime

        # Save the processed data for this gaze type
        np.save(f'{output_dir}/{subject}_{session}_{task}_{run}_contrastgrid_{gaze_type}.npy', stim_acrosstime)
        print(f'Completed processing for {subject} {session} {task} {run} {gaze_type}')

if __name__ == "__main__":
    import sys

    sys.path.append('/DATA/publish/mocet/analysis/scripts')
    from utils.base import get_minecraft_subjects, get_project_directory, get_configs

    subject_pool = get_minecraft_subjects()
    project_dir = get_project_directory()
    configs = get_configs()

    task = configs['task']
    finalTR = configs['interval']

    gaze_types = ['mocet', 'polynomial', 'linear', 'uncorrected']
    #gaze_types = ['linear']

    usable_data = pickle.load(open('../../data/usable_data_list.pkl', 'rb'))
    subjects = []
    for key in list(usable_data.keys()):
        subjects.append(key[0])
    subjects = list(set(subjects))
    subjects.sort()

    padding_width, padding_height = 3200, 3800
    crop_width, crop_height = 3200, 3200
    imageres_row = 64
    imageres_col = 64
    npx = 5  # 5 # pixel per grid width (or height)
    addpx = 1600
    n_TRs = 510
    n_movies = 4 # number of movies to process in parallel

    keys = []
    for subject in subjects:
        sessions = subject_pool[subject].keys()
        for session in sessions:
            runs = subject_pool[subject][session]
            for r in runs:
                run = f'run-{r}'
                np.random.seed(0)
                key = (subject, session, task, run)
                if key in usable_data.keys():
                    keys.append(key)

    while True:
        movie_data = {}
        movie_keys = []
        while len(movie_data) < int(n_movies*len(gaze_types)) and len(keys) > 0:
            subject, session, task, run = keys[0]
            output_dir = f'../../data/contrastgrid/{subject}/{session}'
            file_conditions = [False] * len(gaze_types)
            for g, gaze_type in enumerate(gaze_types):
                if os.path.exists(f'{output_dir}/{subject}_{session}_{task}_{run}_contrastgrid_{gaze_type}.npy'):
                    file_conditions[g] = True
            if all(file_conditions):
                print(f'Contrast grid for {subject} {session} {task} {run} already exists. Skipping...')
                keys.pop(0)
            else:
                frames_loaded, fps_loaded, key = load_movie(keys.pop(0))
                if frames_loaded is not None:
                    combined_keys = [(gaze_type, )+key for gaze_type in gaze_types]
                    for combined_key in combined_keys:
                        movie_data[combined_key] = (frames_loaded, fps_loaded)
                        movie_keys.append(combined_key)
        print(f'Loaded {len(movie_data)/len(gaze_types)} movies')

        time_sta = time.time()
        n_processes = 12
        with Pool(n_processes) as pool:
            pool.starmap(calculate_contrastgrid, [key for key in movie_keys])
        print(f'Elapsed time {(time.time() - time_sta):3.3f}')

        if len(keys) == 0:
            print('All data processed.')
            break
        else:
            print(f'Remaining data: {len(keys)}')