import numpy as np


class Sensor:
    def __init__(self, mat, sensor, data, labels, interp_funcs):
        self._data = data[f"mat_{mat}"][sensor]
        self._labels = labels[f"mat_{mat}"]
        self._interp_funcs = interp_funcs[f"mat_{mat}"][sensor]

        self._raw_cls_data_list = self._build_raw_cls_data_list()

    def _build_raw_cls_data_list(self):
        cls_data_list = []
        for label in self._labels:
            start = label["start"]
            end = label["end"]
            cls = label["label"]
            target = label["target"]
            heater_data_list = []
            for i in range(10):
                heater_step = self._data[i]
                time_data = heater_step["Time Since PowerOn"].values
                filt_data = heater_step["Filtered"].values

                mask = (time_data >= start) & (time_data <= end)
                time_cls = time_data[mask]
                filt_cls = filt_data[mask]
                heater_data_list.append({
                    "start": time_cls[0],
                    "end": time_cls[-1],
                    "num_samples": len(time_cls),
                    "sample_times": time_cls,
                    "sample_vals": filt_cls
                })

            cls_data_list.append({
                "class": cls,
                "target": target,
                "start": start,
                "end": end,
                "heater_data_list": heater_data_list
            })

        return cls_data_list

    def get_interpolated_data(self, force_num_samples: int = None):
        cls_data_list = []
        for raw_cls_data in self._raw_cls_data_list:
            cls = raw_cls_data["class"]
            target = raw_cls_data["target"]
            start = raw_cls_data["start"]
            end = raw_cls_data["end"]
            heater_data_list = raw_cls_data["heater_data_list"]

            if force_num_samples:
                num_samples = force_num_samples
            else:
                num_samples = max([el["num_samples"]
                                  for el in heater_data_list])

            sample_times = np.linspace(start, end, num_samples)
            interp_heater_data_list = []
            for i in range(10):
                interp_heater_data = self._interp_funcs[i](sample_times)
                interp_heater_data_list.append(interp_heater_data)

            cls_data_list.append({
                "class": cls,
                "target": target,
                "start": start,
                "end": end,
                "time_arr": sample_times,
                "X": np.array(interp_heater_data_list),
                "y": np.array([cls] * num_samples, dtype=np.int32),
                "targets": np.array([target] * num_samples, dtype=np.float32)
            })

        return cls_data_list
