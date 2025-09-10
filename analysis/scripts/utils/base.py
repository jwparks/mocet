def get_configs():
    configs = {}
    configs['calibration_onsets'] = [1, 494]
    configs['calibration_points'] = [24, 12]
    configs['interval'] = 1.6
    configs['task_duration'] = 816
    configs['task'] = 'task-mcHERDING'

    configs['calibration_offset_start'] = 0.55
    configs['calibration_offset_end'] = -0.55
    configs['calibration_threshold'] = 1.0
    configs['px_per_deg'] = 78.0487
    configs['avg_pupil_diameter_mm'] = 5

    configs['calibration_coordinates'] = [[200, 166], [200, 500], [200, 833],
                                          [600, 166], [600, 500], [600, 833],
                                          [1000, 166], [1000, 500], [1000, 833],
                                          [1400, 166], [1400, 500], [1400, 833]]

    configs['calibration_order'] = [4, 11, 6, 2, 7, 0, 10, 5, 9, 8, 1, 3]
    return configs


def get_project_directory():
    """
    Returns the root directory of the project.
    """
    return '/DATA/publish/mocet/analysis'


def get_minecraft_subjects():
    subject_pool = {
        'sub-003': {'ses-07R': ([1, 2, 3, 4, 5]),
                    'ses-13R': ([1, 2, 3, 4, 5, 6, 7])},
        'sub-004': {'ses-07R': ([1, 2, 3, 4, 5, 6]),
                    'ses-13': ([1, 2, 3, 4, 5, 6, ])},
        'sub-005': {'ses-07A': ([1, 2, 3, 4, 5, 6]),
                    'ses-13': ([1, 2, 3, 4, 5, 6])},
        'sub-006': {'ses-07R': ([1, 2, 3, 4, 5, 6]),
                    'ses-13': ([1, 2, 3, 4, 5, 6])},
        'sub-008': {'ses-07R': ([1, 2, 3, 4, 5, 6]),
                    'ses-13': ([1, 2, 3, 4, 5, 6])},
        'sub-009': {'ses-07': ([1, 2, 3, 4, 5, 6]),
                    'ses-13': ([1, 2, 3, 4, 5, 6])},
        'sub-010': {'ses-07': ([1, 2, 3, 4, 5, 6]),
                    'ses-13': ([1, 2, 3, 4, 5, 6])},
        'sub-011': {'ses-07': ([1, 2, 3, 4, 5, 6]),
                    'ses-13': ([1, 2, 3, 4, 5, 6])},
        'sub-015': {'ses-07': ([1, 2, 3, 4, 5, 6]),
                    'ses-13': ([1, 2, 3, 4, 5, 6])},
        'sub-016': {'ses-07': ([1, 2, 3, 4, 5, 6]),
                    'ses-13': ([1, 2, 3, 4, 5, 6])},
        'sub-018': {'ses-07': ([1, 2, 3, 4, 5, 6]),
                    'ses-13': ([1, 2, 3, 4, 5, 6])},
        'sub-019': {'ses-07': ([1, 2, 3, 4]),
                    'ses-07A': ([1, 2, 3, 4]),
                    'ses-13': ([1, 2, 3, 4])},
        'sub-020': {'ses-07': ([1, 2, 3, 4, 5, 6]),
                    'ses-13': ([1, 2, 3, 4, 5, 6])},
        'sub-021': {'ses-07': ([1, 2, 3, 4, 5, 6]),
                    'ses-13': ([1, 2, 3, 4, 5, 6])},
        'sub-022': {'ses-07': ([1, 2, 3, 4, 5, 6]),
                    'ses-13': ([1, 2, 3, 4, 5, 6])},
        'sub-023': {'ses-07': ([1, 2, 3, 4, 5, 6]),
                    'ses-13': ([1, 2, 3, 4, 5, 6])},
        'sub-024': {'ses-07': ([1, 2, 3, 4, 5, 6]),
                    'ses-13': ([1, 2, 3, 4, 5]),
                    'ses-13A': ([1])},
        'sub-PBJ': {'ses-07': ([1, 2, 3, 4, 5]),
                    'ses-13': ([1, 2, 3, 4, 5, 6, 7])}
    }
    return subject_pool


def get_mm_px_scaling():
    """
    Returns the pupil scaling factor.
    measured_pfh_in_mm: measured palpebral fissure vertical height(pfh) in mm.
                        if not available (None), use the average pfh value across subjects.
    measured_pfh_in_px: measured palpebral fissure vertical height(pfh) in px (from eyetracker).
    """
    measured_pfh_in_mm = {'sub-003': 11, 'sub-004': None, 'sub-005': None, 'sub-006': 9.5, 'sub-008': 12,
                          'sub-009': 9, 'sub-010': None, 'sub-011': 9, 'sub-015': None, 'sub-016': 12.5,
                          'sub-018': 12, 'sub-019': None, 'sub-020': 12, 'sub-021': 10.5,
                          'sub-022': 10.5, 'sub-023': 11, 'sub-024': 12, 'sub-PBJ': 8.5}

    measured_pfh_in_px = {
        'sub-003': {'ses-07R': 165.4, 'ses-13R': 153.2},
        'sub-004': {'ses-07R': 129.0, 'ses-13': 109.2},
        'sub-005': {'ses-07A': 80.1, 'ses-13': 89.6},
        'sub-006': {'ses-07R': 95.6, 'ses-13': 103.1},
        'sub-008': {'ses-07R': 118.1, 'ses-13': 120.8},
        'sub-009': {'ses-07': 113.8, 'ses-13': 140.3},
        'sub-010': {'ses-07': 148.2, 'ses-13': 131.0},
        'sub-011': {'ses-07': 136.1, 'ses-13': 117.2},
        'sub-015': {'ses-07': 111.9, 'ses-13': 113.6},
        'sub-016': {'ses-07': 128.4, 'ses-13': 130.4},
        'sub-018': {'ses-07': 137.7, 'ses-13': 120.1},
        'sub-019': {'ses-07': 124.0, 'ses-07A': 114.6, 'ses-13': 97.2},
        'sub-020': {'ses-07': 98.0, 'ses-13': 124.8},
        'sub-021': {'ses-07': 90.6, 'ses-13': 114.1},
        'sub-022': {'ses-07': 112.5, 'ses-13': 119.3},
        'sub-023': {'ses-07': 148.9, 'ses-13': 131.5},
        'sub-024': {'ses-07': 127.0, 'ses-13': 128.3, 'ses-13A': 103.7},
        'sub-PBJ': {'ses-07': 132.2, 'ses-13': 135.5}}

    # Replace None values with average pfh in mm
    non_measured_pfh_in_mm = [v for v in measured_pfh_in_mm.values() if v is not None]
    average_pfh_mm = sum(non_measured_pfh_in_mm) / len(non_measured_pfh_in_mm)
    std_pfh_mm = (sum((x - average_pfh_mm) ** 2 for x in non_measured_pfh_in_mm) / len(non_measured_pfh_in_mm)) ** 0.5
    print('Average palpebral fissure height in mm:', average_pfh_mm, '±', std_pfh_mm)
    for sub, pfh in measured_pfh_in_mm.items():
        if pfh is None:
            measured_pfh_in_mm[sub] = average_pfh_mm

    # Calculate scaling factor for each subject
    scaling_factors = {}
    for sub, pfh_mm in measured_pfh_in_mm.items():
        if sub in measured_pfh_in_px:
            for ses, pfh_px in measured_pfh_in_px[sub].items():
                scaling_factors[(sub, ses)] = pfh_mm / pfh_px  # Calculate scaling factor in mm/px
    return scaling_factors  # Default scaling factor, can be adjusted based on calibration