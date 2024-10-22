import numpy as np
import numpy.typing as npt
from sensor import Sensor


class Dataset:
    def __init__(self, sensor_data, labels, interp_funcs):
        self._sensor_data = sensor_data
        self._labels = labels
        self._interp_funcs = interp_funcs
        self._sensor_classes = {}
        self._build_sensor_classes()

    def _build_sensor_classes(self):
        for mat in range(2):
            self._sensor_classes[f"mat_{mat}"] = []
            for sensor in range(8):
                s = Sensor(mat,
                           sensor,
                           self._sensor_data,
                           self._labels,
                           self._interp_funcs)

                self._sensor_classes[f"mat_{mat}"].append(s)

    def get_sensor_cls(self, mat, sensor, num_samples=None, as_log=False):
        """
        returns X, y, time array, targets array
        """
        s: Sensor = self._sensor_classes[f"mat_{mat}"][sensor]
        interpolated_data = s.get_interpolated_data(
            force_num_samples=num_samples)
        X = np.array([[]] * 10)
        y = np.array([], dtype=np.int32)
        time_arr = np.array([])
        targets = np.array([])
        for cls_data in interpolated_data:
            X = np.append(X, cls_data["X"], axis=1)
            y = np.append(y, cls_data["y"])
            time_arr = np.append(time_arr, cls_data["time_arr"])
            targets = np.append(targets, cls_data["targets"])
        if as_log:
            X = np.log(X)
        return X.T, y, time_arr, targets

    def get_sensor_pair_cls(self, mat, sensor_pair,
                            num_samples=None,
                            as_log=False,
                            as_mean=False,
                            sort_by_class=False):
        """
        returns X, y, time array, targets
        """
        if len(sensor_pair) != 2:
            raise Exception("sensors_list must contain exactly 2 sensor ids!")

        s1: Sensor = self._sensor_classes[f"mat_{mat}"][sensor_pair[0]]
        s2: Sensor = self._sensor_classes[f"mat_{mat}"][sensor_pair[1]]

        s1_data = s1.get_interpolated_data(force_num_samples=num_samples)
        s2_data = s2.get_interpolated_data(force_num_samples=num_samples)
        if sort_by_class:
            s1_data = sorted(s1_data, key=lambda d: d["class"])
            s2_data = sorted(s2_data, key=lambda d: d["class"])

        if as_mean:
            X = np.array([[]] * 10)
        else:
            X = np.array([[]] * 20)

        y = np.array([], dtype=np.int32)
        time_arr = np.array([])
        targets = np.array([])

        for cls_data_1, cls_data_2 in zip(s1_data, s2_data):
            if not (cls_data_1["y"] == cls_data_2["y"]).all():
                raise Exception(f"Classes are not the same!")

            if not (cls_data_1["time_arr"] == cls_data_2["time_arr"]).all():
                raise Exception(f"Time arrays are not the same!")

            X_1 = cls_data_1["X"]
            X_2 = cls_data_2["X"]
            if as_mean:
                X_1_2 = np.mean(np.array([X_1, X_2]), axis=0)
            else:
                X_1_2 = np.append(X_1, X_2, axis=0)
            X = np.append(X, X_1_2, axis=1)
            y = np.append(y, cls_data_1["y"])
            time_arr = np.append(time_arr, cls_data_1["time_arr"])
            targets = np.append(targets, cls_data_1["targets"])

        if as_log:
            X = np.log(X)

        return X.T, y, time_arr, targets

    def clean_up_regression_data(self, X, targets):
        cond = ~np.isnan(targets)
        X_clean = X[cond]
        y_clean = targets[cond]
        return X_clean, y_clean

    def calibrate_data(self, X_src: npt.NDArray, X_target: npt.NDArray) -> npt.NDArray:
        """
        calibrates X_src to match X_target.

        returns X_src_calibrated.

        X_src = [10, 14, 20]

        X_target = [30, 38, 50]

        -> X_src_calibrated = [30, 38, 20]
        """
        interval_src = np.std(X_src, axis=0)
        interval_target = np.std(X_target, axis=0)
        it_over_is = interval_target / interval_src
        X_scr_calibrated = X_src.copy()
        X_scr_calibrated = X_scr_calibrated * it_over_is
        mean_diff = np.mean(X_scr_calibrated, axis=0) - \
            np.mean(X_target, axis=0)
        X_scr_calibrated = X_scr_calibrated - mean_diff
        return X_scr_calibrated


def plot_data_pair(X_l, X_r, time_arr, title):
    data_T_l = X_l.T
    data_T_r = X_r.T
    fig = go.Figure()

    for i, data in enumerate(data_T_l):
        fig.add_trace(go.Scatter(x=time_arr,
                                 y=data,
                                 mode="markers",
                                 name=f"L {i}"))
    for i, data in enumerate(data_T_r):
        fig.add_trace(go.Scatter(x=time_arr,
                                 y=data,
                                 mode="markers",
                                 name=f"R {i}"))

    fig.update_layout(title=title, title_x=0.5, width=1000, height=600)
    fig.update_traces(marker=dict(size=2))
    fig.show()


if __name__ == "__main__":
    import pickle
    import plotly.graph_objects as go
    with open("lpf_sensor_data.pkl", "rb") as f:
        sensor_data = pickle.load(f)

    with open("sensor_labels.pkl", "rb") as f:
        labels = pickle.load(f)

    with open("interpolation_functions.pkl", "rb") as f:
        interp_funcs = pickle.load(f)

    dataset = Dataset(sensor_data, labels, interp_funcs)
    X, y, time_arr, targets = dataset.get_sensor_pair_cls(
        0,
        (2, 3),
        num_samples=100,
        as_log=True,
        as_mean=False)
    X_src, X_target = X[:, :10], X[:, 10:]
    X_src_calibrated = dataset.calibrate_data(
        X_src, X_target)
    # diff = np.sum(np.abs(X_src_calibrated - X_target))
    # plot_data_pair(X_src_calibrated, X_target, time_arr, "")
