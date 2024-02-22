import json
import math
import numpy as np
import random
import os

random.seed(1)

class DataProcessor():
    def __init__(self):
        self.max_vals = {}
        self.max_intensity = 145829584.0
        self.max_mz = 993.708252
        self.max_rt = 1471.1

    def process_for_rnn(self, 
                        only_peak=True,
                        min_duration=0,
                        false_prune_percent=0,
                        log_normalize=True,
                        min_max_normalize=True,
                        smooth=0,
                        create_new_json=False,
                        add_derivs=True):
        if log_normalize:
            self.max_intensity = self.max_intensity = 18.79794926504905

        file_str = f"RNN_{only_peak}_{min_duration}_{false_prune_percent}_{log_normalize}_{min_max_normalize}_{smooth}_{add_derivs}.json"
        cur_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(cur_path, "Processed_Data", file_str)

        if os.path.exists(file_path):
            self.load_data(file_path)
            self.train_test_split()
            return self.data, self.test_mzrts
        
        print("loading base data...")
        self.load_data(os.path.join(cur_path, "Data", "EIC14_data.json"))
        print("masking arrays...")
        self.mask_arrays(only_peak=only_peak)
        if smooth > 0:
            print("smoothing truth values...")
            self.smooth_truth_arrays(val=smooth)
        print("padding arrays...")
        self.pad_arrays(side="right")
        if min_duration > 0:
            print("filtering by peak duration...")
            self.filter_by_peak_duration(min_duration)
        if false_prune_percent > 0:
            print("filtering out false peaks...")
            self.filter_false_peaks(false_prune_percent)
        if log_normalize:
            print("log normalizing...")
            self.log_normalize_intensities()
        if min_max_normalize:
            print("min max normalizing...")
            self.min_max_normalize()
        if add_derivs:
            self.add_derivatives()

        print("splitting into test train...")
        self.train_test_split()

        if create_new_json:
            print("creating new json...")
            with open(file_path, "w") as fout:
                json.dump(self.data, fout, indent=1)
                fout.close()
        print("processing complete. ")
        
        return self.data, self.test_mzrts 


    def load_data(self, path_str):
        with open(path_str, "r") as f:
            data = json.load(f)
            f.close()
        self.data = data
    
    def process_for_cnn(self, only_peak=True, min_duration=0, false_prune_percent=0, log_normalize=True, min_max_normalize=True, size=32):
        file_str = f"CNN_{only_peak}_{min_duration}_{false_prune_percent}_{log_normalize}_{min_max_normalize}_{size}"
        cur_path = os.path.abspath(__file__)
        file_path = os.path.join(cur_path, "Processed_Data", file_str)

        if os.path.exists(file_path):
            self.load_data(file_path)
            self.train_test_split()
            return self.data, self.test_mzrts
        
        self.load_data("Data/EIC_14_data.json", "r")
        self.mask_arrays(only_peak=only_peak)
        if min_duration > 0:
            self.filter_by_peak_duration(min_duration)
        if false_prune_percent > 0:
            self.filter_false_peaks(false_prune_percent)
        if log_normalize:
            self.log_normalize_intensities()
        if min_max_normalize:
            self.min_max_normalize()
        self.inputs_for_each_timestep(size=size)

        self.train_test_split()

        return self.data, self.test_mzrts

    def mask_arrays(self, only_peak=False):
        for mzml_key in self.data.keys():
            for mzrt_key in self.data[mzml_key].keys():
                self.data[mzml_key][mzrt_key]["mask"] = []
                for rt in self.data[mzml_key][mzrt_key]["rt array"]:
                    if not only_peak:
                        self.data[mzml_key][mzrt_key]["mask"].append(1)
                    elif self.data[mzml_key][mzrt_key]["rt start"] < rt < self.data[mzml_key][mzrt_key]["rt end"]:
                        self.data[mzml_key][mzrt_key]["mask"].append(1)
                    else:
                        self.data[mzml_key][mzrt_key]["mask"].append(0)
    
    def smooth_truth_arrays(self, val):
        for mzml_key in self.data.keys():
            for mzrt_key in self.data[mzml_key].keys():
                for i in range(len(self.data[mzml_key][mzrt_key]["truth array"])):
                    if self.data[mzml_key][mzrt_key]["truth array"][i] == 1:
                        self.data[mzml_key][mzrt_key]["truth array"][i] -= val
                    else:
                        self.data[mzml_key][mzrt_key]["truth array"][i] += val

    def pad_arrays(self, n=51, side="right"):
        for mzml_key in self.data.keys():
            for mzrt_key in self.data[mzml_key].keys():
                l = len(self.data[mzml_key][mzrt_key]["rt array"])
                pad = n - l
                if side == "right":
                    self.data[mzml_key][mzrt_key]["rt array"] = self.data[mzml_key][mzrt_key]["rt array"] + [0]*pad
                    self.data[mzml_key][mzrt_key]["intensity array"] = self.data[mzml_key][mzrt_key]["intensity array"] + [0]*pad
                    self.data[mzml_key][mzrt_key]["mz array"] = self.data[mzml_key][mzrt_key]["mz array"] + [0]*pad
                    self.data[mzml_key][mzrt_key]["mask"] = self.data[mzml_key][mzrt_key]["mask"] + [0]*pad
                    self.data[mzml_key][mzrt_key]["truth array"] = self.data[mzml_key][mzrt_key]["truth array"] + [0]*pad
                elif side == "left":
                    self.data[mzml_key][mzrt_key]["rt array"] = [0]*pad + self.data[mzml_key][mzrt_key]["rt array"]
                    self.data[mzml_key][mzrt_key]["intensity array"] = [0]*pad + self.data[mzml_key][mzrt_key]["intensity array"]
                    self.data[mzml_key][mzrt_key]["mz array"] = [0]*pad + self.data[mzml_key][mzrt_key]["mz array"]
                    self.data[mzml_key][mzrt_key]["mask"] = [0]*pad + self.data[mzml_key][mzrt_key]["mask"]
                    self.data[mzml_key][mzrt_key]["truth array"] = [0]*pad + self.data[mzml_key][mzrt_key]["truth array"]
                
    def filter_by_peak_duration(self, duration):
        for mzml_key in self.data.keys():
            to_delete = []
            for mzrt_key in self.data[mzml_key].keys():
                if self.data[mzml_key][mzrt_key]["rt end"] - self.data[mzml_key][mzrt_key]["rt start"] < duration:
                    to_delete.append(mzrt_key)
            for d in to_delete:
                del self.data[mzml_key][d]

    def filter_false_peaks(self, percentage):
        for mzml_key in self.data.keys():
            to_delete = []
            for mzrt_key in self.data[mzml_key].keys():
                if self.data[mzml_key][mzrt_key]["truth"] == 0 and random.randint(0, 100) < percentage:
                    to_delete.append(mzrt_key)
            for d in to_delete:
                del self.data[mzml_key][d]

    def log_normalize_intensities(self):
        for mzml_key in self.data.keys():
            for mzrt_key in self.data[mzml_key].keys():
                for i in range(len(self.data[mzml_key][mzrt_key]["intensity array"])):
                    if self.data[mzml_key][mzrt_key]["intensity array"][i] == 0:
                        self.data[mzml_key][mzrt_key]["intensity array"][i] += 0.0000001
                self.data[mzml_key][mzrt_key]["intensity array"] = list(np.log(self.data[mzml_key][mzrt_key]["intensity array"]))

    def min_max_normalize(self):
        for mzml_key in self.data.keys():
            for mzrt_key in self.data[mzml_key].keys():
                for i in range(len(self.data[mzml_key][mzrt_key]["intensity array"])):
                    self.data[mzml_key][mzrt_key]["intensity array"][i] = self.data[mzml_key][mzrt_key]["intensity array"][i]/self.max_intensity
                    self.data[mzml_key][mzrt_key]["mz array"][i] = self.data[mzml_key][mzrt_key]["mz array"][i]/self.max_mz
                    self.data[mzml_key][mzrt_key]["rt array"][i] = self.data[mzml_key][mzrt_key]["rt array"][i]/self.max_rt
        
        self.max_vals = {"max_intensity" : self.max_intensity, "max_mz" : self.max_mz, "max_rt" : self.max_rt}

    def add_derivatives(self):
        for mzml_key in self.data.keys():
            for mzrt_key in self.data[mzml_key].keys():
                derivs = [0]
                for i in range(len(self.data[mzml_key][mzrt_key]["intensity array"])-1):
                    derivs.append(self.data[mzml_key][mzrt_key]["intensity array"][i+1]-self.data[mzml_key][mzrt_key]["intensity array"][i])
                self.data[mzml_key][mzrt_key]["derivative array"] = derivs

    
    def inputs_for_each_timestep(self, size=32):
        for mzml_key in self.data.keys():
            for mzrt_key in self.data[mzml_key].keys():
                new_peaks = []
                for i in range(len(self.data[mzml_key][mzrt_key]["mask"])):
                    if self.data[mzml_key][mzrt_key]["mask"][i] == 1:
                        end = i+1 if i < len(self.data[mzml_key][mzrt_key]["mask"]) else None
                        pad = size - end
                        new_peaks.append(
                            {
                                "intensity array" : pad*[0] + self.data[mzml_key][mzrt_key]["intensity array"][:end],
                                "mz array" : pad*[0] + self.data[mzml_key][mzrt_key]["mz array"][:end],
                                "rt array" : pad*[0] + self.data[mzml_key][mzrt_key]["rt array"][:end],
                                "truth array" : pad*[0] + self.data[mzml_key][mzrt_key]["truth array"][:end],
                                "mask" : pad*[0] + self.data[mzml_key][mzrt_key]["mask"][:end]
                            }
                        )
                self.data[mzml_key][mzrt_key] = new_peaks





    # add way to change from 20% to something else maybe
    def train_test_split(self):
        total_peaks = 0
        total_true = 0
        total_false = 0
        mzrt_peak_count = {}
        test_mzrts = set()
        #loop through and count peaks in each mzrt group
        for mzml_key in self.data.keys():
            for mzrt_key in self.data[mzml_key].keys():
                peak_count = 1 if type(self.data[mzml_key][mzrt_key]) == type({}) else len(self.data[mzml_key][mzrt_key])
                total_peaks += peak_count
                if self.data[mzml_key][mzrt_key]["truth"] == 1:
                    total_true += sum(self.data[mzml_key][mzrt_key]["mask"])
                else:
                    total_false += sum(self.data[mzml_key][mzrt_key]["mask"])
                if mzrt_key in mzrt_peak_count:
                    mzrt_peak_count[mzrt_key] += peak_count
                else:
                    mzrt_peak_count[mzrt_key] = peak_count
        
        #divide into train and test
        num_test_peaks = 0
        mzrt_key_list = list(mzrt_peak_count)
        while num_test_peaks < total_peaks//5:
            new_test_mzrt = mzrt_key_list.pop(random.randint(0, len(mzrt_key_list)-1))
            num_test_peaks += mzrt_peak_count[new_test_mzrt]
            del mzrt_peak_count[new_test_mzrt]
            test_mzrts.add(new_test_mzrt)
        self.test_mzrts = test_mzrts
        print("true:", total_true, "false:", total_false, "true%:", total_true/(total_true+total_false))
        