import numpy as np


class Sensor:
    def __init__(self, matrix, sensor_idx, interp_funcs, labels):
        self.interp_funcs = interp_funcs[f"mat_{matrix}"][sensor_idx]
        self.labels = labels[f"mat_{matrix}"]
        self.num_s = 100

    def get_data_for_region(self, region_idx):
        region = self.labels[region_idx]
        Y = []
        t = np.linspace(region["start"], region["end"], self.num_s)
        for hp in range(10):
            y = np.log(self.interp_funcs[hp](t))
            Y.append(y)
        return {
            "y": np.array(Y),  # sensor data
            "label": region["label"],  # clf label
            "reg": region["target"],  # regression target
            "t": t,  # time
        }

    def get_ml_data_for_region(self, region_idx):
        region_data = self.get_data_for_region(region_idx)
        X = region_data["y"]
        y = np.array([region_data["label"]] * self.num_s, dtype=np.int32)
        r = np.array([region_data["reg"]] * self.num_s)
        t = region_data["t"]
        return X.T, y, r, t

    def get_ml_data_list(self):
        X = np.array([[]] * 10)
        y = np.array([], dtype=np.int32)
        r = np.array([])
        t = np.array([])
        for i in range(len(self.labels)):
            region_data = self.get_data_for_region(i)
            X = np.append(X, region_data["y"], axis=1)
            y = np.append(y, np.array([region_data["label"]] * self.num_s))
            r = np.append(r, np.array([region_data["reg"]] * self.num_s))
            t = np.append(t, region_data["t"])
        return X.T, y, r, t  # sensor data, clf label, reg target, time


def get_sensor_tuple_data(matrix, s_l_idx, s_r_idx, interp_funcs, labels):
    """ 
    Returns: Sensor data, Clf labels, Reg data, Time arr
    """
    s_l = Sensor(matrix, s_l_idx, interp_funcs, labels)
    X_l, y_l, r_l, t_l = s_l.get_ml_data_list()
    s_r = Sensor(matrix, s_r_idx, interp_funcs, labels)
    X_r, y_r, r_r, t_r = s_r.get_ml_data_list()

    if not (y_l == y_r).all():
        raise Exception(f"Classes are not the same!")
    if not (r_l == r_r).all():
        raise Exception(f"Reg targets are not the same!")
    if not (t_l == t_r).all():
        raise Exception(f"Time data are not the same!")

    X_tuple = np.concatenate((X_l, X_r), axis=1)
    return X_tuple, y_l, r_l, t_l  # Sensor data, Clf labels, Reg data, Time arr


def get_sensor_tuple_data_for_region(matrix, s_l_idx, s_r_idx, interp_funcs, labels, region_idx):
    s_l = Sensor(matrix, s_l_idx, interp_funcs, labels)
    X_l, y_l, r_l, t_l = s_l.get_ml_data_for_region(region_idx)
    s_r = Sensor(matrix, s_r_idx, interp_funcs, labels)
    X_r, y_r, r_r, t_r = s_r.get_ml_data_for_region(region_idx)

    if not (y_l == y_r).all():
        raise Exception(f"Classes are not the same!")
    if not (r_l == r_r).all():
        raise Exception(f"Reg targets are not the same!")
    if not (t_l == t_r).all():
        raise Exception(f"Time data are not the same!")

    X_tuple = np.concatenate((X_l, X_r), axis=1)
    return X_tuple, y_l, r_l, t_l  # Sensor data, Clf labels, Reg data, Time arr
