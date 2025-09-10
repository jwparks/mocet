import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from scipy.linalg import LinAlgWarning
from scipy.interpolate import Rbf
from scipy import interpolate
from sklearn.linear_model import LinearRegression
from numpy.polynomial.legendre import Legendre
from mocet import utils
from mocet import simulation
import warnings

def make_poly_regressors(n_samples, order=2):
    X = np.ones((n_samples, 1))
    for d in range(order):
        poly = Legendre.basis(d + 1)
        poly_trend = poly(np.linspace(-1, 1, n_samples))
        X = np.hstack((X, poly_trend[:, None]))
    return X

def get_motion_params(motion_params_fname, source='fmriprep', large_motion_params=False, use_mm_deg = False):
    if source == 'fmriprep':
        if large_motion_params:
            motion_param_labels = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                                   'trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1',
                                   'rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1',
                                   'trans_x_power2', 'trans_y_power2', 'trans_z_power2',
                                   'rot_x_power2', 'rot_y_power2', 'rot_z_power2',
                                   'trans_x_derivative1_power2', 'trans_y_derivative1_power2',
                                   'trans_z_derivative1_power2',
                                   'rot_x_derivative1_power2', 'rot_y_derivative1_power2', 'rot_z_derivative1_power2']
        else:
            motion_param_labels = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']

        fmriprep_confounds = pd.read_csv(motion_params_fname, delimiter='\t')
        motion_params = fmriprep_confounds[motion_param_labels]
        motion_params = np.nan_to_num(motion_params)
        if use_mm_deg and not large_motion_params:
            motion_params[3:] = np.rad2deg(motion_params[3:])
        motion_params = np.nan_to_num(motion_params)
        motion_params = motion_params - motion_params[0, :]
        motion_params = np.vstack([motion_params, motion_params[-1, :]])
        motion_params = -motion_params
        return motion_params
    elif source == 'fsl' or source == 'mcflirt':
        if large_motion_params:
            raise(ValueError("FSL motion parameters do not support large motion parameters."))
        motion_params = np.genfromtxt(motion_params_fname)
        rotations = motion_params[:, :3] # in radians
        translations = motion_params[:, 3:6] # in mm
        if use_mm_deg:
            rotations = np.rad2deg(rotations)
        motion_params = np.hstack((translations, rotations))
        return motion_params

def apply_mocet(pupil_data,
                motion_params=None,
                motion_params_fname=None,
                pupil_confidence=None,
                confidence_threshold=0.75,
                motion_source='fmriprep',
                large_motion_params=False,
                polynomial_order=0,
                interactions=False,
                return_weights = False,
                use_mm_deg = False):
    if motion_params is None and motion_params_fname is None:
        raise(ValueError("Either motion_params or motion_params_fname must be provided."))
    if motion_params is not None and motion_params_fname is not None:
        warnings.warn("Both motion_params and motion_params_fname are provided. motion_params will be used.")
    if motion_params is None and motion_params_fname is not None:
        if motion_source not in ['fmriprep', 'fsl', 'mcflirt']:
            raise(ValueError("motion_source must be either 'fmriprep', 'fsl', or 'mcflirt'."))
        motion_params = get_motion_params(motion_params_fname,
                                          source=motion_source,
                                          large_motion_params=large_motion_params,
                                          use_mm_deg=use_mm_deg)
    if motion_params.shape[0] < 2:
        raise(ValueError("motion_params must have at least 2 time points."))

    X = np.zeros((len(pupil_data), motion_params.shape[1]))
    x = np.arange(0, len(motion_params))
    for i in range(motion_params.shape[1]):
        y = motion_params[:, i]
        f = interpolate.interp1d(x, y)
        xnew = np.linspace(0, len(motion_params) - 1, len(pupil_data))
        X[:, i] = f(xnew)

    if interactions:
        from itertools import combinations
        # including interaction terms through combination of 6 parameters
        if large_motion_params:
            for f1, f2 in combinations(range(24), 2):
                X = np.hstack((X, (X[:, f1] * X[:, f2])[:, None]))
        else:
            for f1, f2 in combinations(range(6), 2):
                X = np.hstack((X, (X[:, f1] * X[:, f2])[:, None]))
    if polynomial_order != 0:
        X = np.hstack((X, make_poly_regressors(len(X), order=polynomial_order)))

    if pupil_confidence is None:
        coefs_ = []
        dedrift_regressor = np.zeros((len(pupil_data), 2))
        for i in range(2):
            reg = LinearRegression(fit_intercept=False).fit(X, pupil_data[:, i])
            coefs_.append(reg.coef_)
            dedrift_regressor[:, i] = reg.predict(X)
        pupil_data = pupil_data[:, :2] - dedrift_regressor
    else:
        coefs_ = []
        dedrift_regressor = np.zeros((len(pupil_data), 2))
        for i in range(2):
            reg = LinearRegression(fit_intercept=False).fit(X[pupil_confidence>=confidence_threshold, :],
                                                            pupil_data[pupil_confidence>=confidence_threshold, i])
            coefs_.append(reg.coef_)
            dedrift_regressor[:, i] = reg.predict(X)
        pupil_data = pupil_data[:, :2] - dedrift_regressor

    if return_weights:
        return pupil_data, coefs_, dedrift_regressor
    else:
        return pupil_data


class EyetrackingCalibration():
    def __init__(self, calibration_coordinates, calibration_order,
                 repeat=True, method='linear', smoothness=0, noise=0.01):
        """
        calibration_coordinates: list of calibration points in the form [[x1, y1], [x2, y2], ...]
        calibration_order: list of indices that define the order in which the calibration points are used
        repeat: if True, the calibration points are repeated in the order defined by calibration_order
        """

        self.calibration_coordinates = calibration_coordinates
        self.calibration_order = calibration_order

        self.calibration_coordinates = np.array(self.calibration_coordinates)
        if repeat:
            self.calibration_coordinates = np.tile(self.calibration_coordinates, [2, 1])
            self.calibration_order = self.calibration_order * 2
        self.calibration_order = np.array(self.calibration_order)

        self.n_calibrations = len(self.calibration_order)
        self.method = method
        self.smoothness = smoothness
        self.repeat = repeat
        self.noise = noise

    def fit(self, calibration_x, calibration_y, allow_lossy_fit=False):
        try:
            self.interpolater(calibration_x, calibration_y,
                              self.calibration_coordinates[self.calibration_order, 0],
                              self.calibration_coordinates[self.calibration_order, 1])
            return True
        except LinAlgError:
            if allow_lossy_fit:
                non_singular_idx = np.sort(np.unique(calibration_x, return_index=True)[1])
                if len(non_singular_idx) < 4:
                    calibration_x = np.array(calibration_x) + np.random.normal(0, self.noise, len(calibration_x))
                    calibration_y = np.array(calibration_y) + np.random.normal(0, self.noise, len(calibration_x))
                    self.interpolater(calibration_x, calibration_y,
                                      self.calibration_coordinates[self.calibration_order, 0],
                                      self.calibration_coordinates[self.calibration_order, 1])
                    print('Calibration points are not sufficient to fit the interpolator due to duplicated points. '
                                  'Therefore, small noise was added to the calibration points to augment the data. '
                                  'This may lead to significantly inaccurate results. ')
                else:
                    calibration_x = np.array(calibration_x)[non_singular_idx]
                    calibration_y = np.array(calibration_y)[non_singular_idx]
                    calibration_point_x = self.calibration_coordinates[self.calibration_order, 0][[non_singular_idx]]
                    calibration_point_y = self.calibration_coordinates[self.calibration_order, 1][[non_singular_idx]]
                    self.interpolater(calibration_x, calibration_y, calibration_point_x, calibration_point_y)
            else:
                return False
        except LinAlgWarning:
            if allow_lossy_fit:
                if self.repeat:
                    calibration_x = np.array(calibration_x)[:int(len(self.calibration_order) / 2)]
                    calibration_y = np.array(calibration_y)[:int(len(self.calibration_order) / 2)]
                    calibration_point_x = self.calibration_coordinates[self.calibration_order, 0][:int(len(self.calibration_order) / 2)]
                    calibration_point_y = self.calibration_coordinates[self.calibration_order, 1][:int(len(self.calibration_order) / 2)]
                    self.interpolater(calibration_x, calibration_y, calibration_point_x, calibration_point_y)
                else:
                    calibration_x = np.array(calibration_x) + np.random.normal(0, self.noise, len(calibration_x))
                    calibration_y = np.array(calibration_y) + np.random.normal(0, self.noise, len(calibration_x))
                    self.interpolater(calibration_x, calibration_y,
                                      self.calibration_coordinates[self.calibration_order, 0],
                                      self.calibration_coordinates[self.calibration_order, 1])
                    print('Calibration points are not sufficient to fit the interpolator due to duplicated points. '
                                  'Therefore, small noise was added to the calibration points to augment the data. '
                                  'This may lead to significantly inaccurate results. ')
            else:
                return False
    def interpolater(self, calibration_x, calibration_y, calibration_point_x, calibration_point_y):
        self.interpolater_x = Rbf(calibration_x,
                                  calibration_y,
                                  calibration_point_x,
                                  function=self.method,
                                  smooth=self.smoothness)
        self.interpolater_y = Rbf(calibration_x,
                                  calibration_y,
                                  calibration_point_y,
                                  function=self.method,
                                  smooth=self.smoothness)

    def transform(self, pupil_data):
        gaze_coordinates = np.zeros(pupil_data.shape)
        gaze_coordinates[:, 0] = self.interpolater_x(pupil_data[:, 0], pupil_data[:, 1])
        gaze_coordinates[:, 1] = self.interpolater_y(pupil_data[:, 0], pupil_data[:, 1])
        return gaze_coordinates

    def reference(self, i):
        ref_x = self.calibration_coordinates[self.calibration_order[i]][0]
        ref_y = self.calibration_coordinates[self.calibration_order[i]][1]
        return ref_x, ref_y
