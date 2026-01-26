import os
import json
import numpy as np

class DataManager:
    def __init__(self, outdir, runner):
        self.outdir = outdir
        self.runner = runner

    def data_exists(self):
        required = [
            "opticalspec.npy",
            "Garr.csv",
            "cavomegaarr.csv",
            "convratio.csv",
        ]
        return all(os.path.exists(os.path.join(self.outdir, f)) for f in required)

    def load_data(self):
        print(">>> Loading existing data")

        data = {
            "opticalspec": np.load(os.path.join(self.outdir, "opticalspec.npy")),
            "Garr": np.loadtxt(os.path.join(self.outdir, "Garr.csv"), delimiter=","),
            "cavomegaarr": np.loadtxt(os.path.join(self.outdir, "cavomegaarr.csv"), delimiter=","),
            "convratio": np.loadtxt(os.path.join(self.outdir, "convratio.csv"), delimiter=","),
        }
        return data

    def get_data(self):
        if self.data_exists():
            return self.load_data()
        else:
            return self.runner.run_simulation()
