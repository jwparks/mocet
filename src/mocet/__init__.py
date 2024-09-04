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

def make_poly_regressors(n_samples, order=2):
    X = np.ones((n_samples, 1))
    for d in range(order):
        poly = Legendre.basis(d + 1)
        poly_trend = poly(np.linspace(-1, 1, n_samples))
        X = np.hstack((X, poly_trend[:, None]))
    return X

def get_motion_params(motion_params_fname, fmriprep=True, large_motion_params=False, use_mm_deg = False):
    if fmriprep:
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
    else:
        raise NotImplementedError('Motion parameters from FSL not implemented yet')


def apply_mocet(pupil_data, motion_params_fname,
                large_motion_params=False,
                polynomial_order=0,
                return_weights = False,
                use_mm_deg = False):
    motion_params = get_motion_params(motion_params_fname,
                                      large_motion_params=large_motion_params,
                                      use_mm_deg=use_mm_deg)

    X = np.zeros((len(pupil_data), motion_params.shape[1]))
    x = np.arange(0, len(motion_params))
    for i in range(motion_params.shape[1]):
        y = motion_params[:, i]
        f = interpolate.interp1d(x, y)
        xnew = np.linspace(0, len(motion_params) - 1, len(pupil_data))
        X[:, i] = f(xnew)
    if polynomial_order != 0:
        X = np.hstack((X, make_poly_regressors(len(X), order=polynomial_order)))

    coefs_ = []
    dedrift_regressor = np.zeros((len(pupil_data), 2))
    for i in range(2):
        reg = LinearRegression().fit(X, pupil_data[:, i])
        coefs_.append(reg.coef_)
        dedrift_regressor[:, i] = reg.predict(X)
    pupil_data = pupil_data[:, :2] - dedrift_regressor
    if return_weights:
        return pupil_data, coefs_, dedrift_regressor
    else:
        return pupil_data


class EyetrackingCalibration():
    def __init__(self, calibration_coordinates=None, calibration_order=None, repeat=True, method='linear', smoothness=0):
        if calibration_coordinates == None:
            self.calibration_coordinates = [[200, 166], [200, 500], [200, 833],
                                            [600, 166], [600, 500], [600, 833],
                                            [1000, 166], [1000, 500], [1000, 833],
                                            [1400, 166], [1400, 500], [1400, 833]]
        else:
            self.calibration_coordinates = calibration_coordinates

        if calibration_order == None:
            self.calibration_order = [4, 11, 6, 2, 7, 0, 10, 5, 9, 8, 1, 3]
        else:
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

    def fit(self, calibration_x, calibration_y):
        try:
            self.interpolater(calibration_x, calibration_y,
                              self.calibration_coordinates[self.calibration_order, 0],
                              self.calibration_coordinates[self.calibration_order, 1])
        except LinAlgError:
            non_singular_idx = np.sort(np.unique(calibration_x, return_index=True)[1])
            calibration_x = np.array(calibration_x)[non_singular_idx]
            calibration_y = np.array(calibration_y)[non_singular_idx]
            calibration_point_x = self.calibration_coordinates[self.calibration_order, 0][[non_singular_idx]]
            calibration_point_y = self.calibration_coordinates[self.calibration_order, 1][[non_singular_idx]]
            self.interpolater(calibration_x, calibration_y, calibration_point_x, calibration_point_y)
        except LinAlgWarning:
            if self.repeat:
                calibration_x = np.array(calibration_x)[:int(len(self.calibration_order) / 2)]
                calibration_y = np.array(calibration_y)[:int(len(self.calibration_order) / 2)]
                calibration_point_x = self.calibration_coordinates[self.calibration_order, 0][:int(len(self.calibration_order) / 2)]
                calibration_point_y = self.calibration_coordinates[self.calibration_order, 1][:int(len(self.calibration_order) / 2)]
                self.interpolater(calibration_x, calibration_y, calibration_point_x, calibration_point_y)



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
