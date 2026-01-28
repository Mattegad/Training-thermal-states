import os
import numpy as np


class DataManager:
    def __init__(self, outdir, runner):
        self.outdir = outdir
        self.runner = runner

    def training_data_exists(self):
        required_training = [
            "opticalspec.npy",
            "Garr.csv",
            "cavomegaarr.csv",
            "convratio.csv",
            "G_pred.csv",
            "moments.csv",
            "G_true.csv",
        ]
        return all(os.path.exists(os.path.join(self.outdir, f)) for f in required_training)
    
    def raw_data_exists(self):
        required_raw = [
            "opticalspec.npy",
            "Garr.csv",
            "cavomegaarr.csv",
            "convratio.csv",
        ]
        return all(os.path.exists(os.path.join(self.outdir, f)) for f in required_raw)

    def load_data(self):
        print(">>> Loading existing data")

        data = {
            "opticalspec": np.load(os.path.join(self.outdir, "opticalspec.npy")),
            "Garr": np.loadtxt(os.path.join(self.outdir, "Garr.csv"), delimiter=","),
            "cavomegaarr": np.loadtxt(os.path.join(self.outdir, "cavomegaarr.csv"), delimiter=","),
            "convratio": np.loadtxt(os.path.join(self.outdir, "convratio.csv"), delimiter=","),
            "G_pred": np.loadtxt(os.path.join(self.outdir, "G_pred.csv"), delimiter=","),
            "moments": np.loadtxt(os.path.join(self.outdir, "moments.csv"), delimiter=","),
            "G_true": np.loadtxt(os.path.join(self.outdir, "G_true.csv"), delimiter=","),
        }
        return data

    def get_data(self):
        if self.training_data_exists():
            return self.load_data()
        elif self.raw_data_exists():
            print(">>> Loading raw data and performing analysis")
            data = {
                "opticalspec": np.load(os.path.join(self.outdir, "opticalspec.npy")),
                "Garr": np.loadtxt(os.path.join(self.outdir, "Garr.csv"), delimiter=","),
                "cavomegaarr": np.loadtxt(os.path.join(self.outdir, "cavomegaarr.csv"), delimiter=","),
                "convratio": np.loadtxt(os.path.join(self.outdir, "convratio.csv"), delimiter=","),
            }
            # Perform analysis
            analysis_results = self.runner.perform_analysis(data)
            data.update(analysis_results)
            return data
        else:
            print(">>> No existing data found, running simulation")
            data = self.runner.run_simulation()
            # Perform analysis
            analysis_results = self.runner.perform_analysis(data)
            data.update(analysis_results)
            return data
