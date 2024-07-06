import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from tabulate import tabulate
class params():
    def __init__(self, s0, k, n, m, m6):
        self.s0 = s0
        self.k = k
        self.n = n
        self.m = m
        self.m6 = m6
        
class helperFunctions():
    def __init__(self, path):
        df = self.import_datafile(path)
        self.DEFAULT_EPS_ARRAY = df.iloc[:240]["E"].astype(float)
        self.DEFAULT_STRAIN_RATES = df["E*"].unique()
        self.CURVES_PER_MATERIAL = len(self.DEFAULT_STRAIN_RATES)
        self.param_list = ["S0", "k", "n", "m", "m6"]
        
    def import_datafile(self, path):
        df = pd.read_csv(path)
        df.columns = ["S0", "k", "n", "m", "E*", "m6", "E", "S"]
        return df
    
    def import_test_datafile(self, path):
        df = pd.read_csv(path)
        df.columns = ["S0", "k", "n", "m", "E*", "m6", "E", "S", "Sf"]
        return df

    def viscoPlasticLaw(self, sigma0, k, n, m, m6, epsilon_dot, epsilon):
        # constants sigma0, k, n, m, m6
        # arrays: epsilon, epsilon_dot

        sigma_pred = sigma0 + k * epsilon**n * epsilon_dot**m * np.exp(m6*epsilon)
        return sigma_pred

    def plotCurves(self, equation, pred, epsilon, real = []):
        plt.scatter(epsilon, pred, label = "predicted", s = 1, c = "green")
        plt.plot(epsilon, equation, label = "Sf")
        if len(real) > 0:
            plt.plot(epsilon, real, label = "S")
        plt.legend()
        plt.show()
        
        
    # Define the viscoplastic law
    def viscoplastic_law_error(self, params, epsilon, epsilon_dot, sigma_exp):
        k, n, m, m6, sigma0 = params
        sigma_pred = self.viscoPlasticLaw(sigma0, k, n, m, m6, epsilon_dot, epsilon)
        return np.mean((np.array(sigma_pred) - np.array(sigma_exp))**2)

    def material_viscoplastic_law_error(self, params, sigma_exp_material):
        error = 0
        for sigma_exp, epsilon_dot in zip(sigma_exp_material, self.DEFAULT_STRAIN_RATES):
            error += self.viscoplastic_law_error(params, self.DEFAULT_EPS_ARRAY, epsilon_dot, sigma_exp)
        return error/self.CURVES_PER_MATERIAL

    def validate(self, epsilon_dot, params, opt_params, real_stress = []):

        predicted_stress = self.viscoPlasticLaw(
            sigma0 = opt_params.s0,
            k = opt_params.k,
            n = opt_params.n,
            m = opt_params.m,
            m6 = opt_params.m6,
            epsilon_dot = epsilon_dot,
            epsilon = self.DEFAULT_EPS_ARRAY
        )

        sf = self.viscoPlasticLaw(
            sigma0 = params.s0,
            k = params.k,
            n = params.n,
            m = params.m,
            m6 = params.m6,
            epsilon_dot = epsilon_dot,
            epsilon = self.DEFAULT_EPS_ARRAY
        )
        self.plotCurves(sf, predicted_stress, self.DEFAULT_EPS_ARRAY, real_stress)
    
    def validate_material(self, df_material, opt_params):
        row1 = df_material.iloc[0]
        real_params = params(row1["S0"], row1["k"], row1["n"], row1["m"], row1["m6"])
        for index, epsilon_dot in enumerate(self.DEFAULT_STRAIN_RATES):
            real_stress = df_material.iloc[index].drop(self.param_list + ["E*"])
            self.validate(epsilon_dot, real_params, opt_params, real_stress)
    
    def addCurveIndex(self, df):
        # Pivot the data, columns strain, values stress, agg mean on the rest
        ci = []
        for i in range(len(df)//240):
             ci.extend([i]*240)
        df["curve_index"] = ci
        return df

    def get_pivot(self, df):
        df["E"] = df["E"].astype(str)
        pivot_df = df.pivot_table(index='curve_index', columns='E', values='S', aggfunc='max')
        pivot_df = pivot_df.drop("0.6025", axis = 1)
        return pivot_df

    def processDfIntoMaterials(self, df):
        # Returns curves with CURVES_PER_MATERIAL (3) strain rates in order, 1 row per curve
        df = self.addCurveIndex(df)
        params_df = df.groupby(["curve_index"]).max().drop(["E", "S"], axis = 1)
        input_params = pd.DataFrame(params_df.pop("E*"))
        pivot_df = self.get_pivot(df)
        df_curves = pd.concat([pivot_df, input_params, params_df], axis = 1)

        group_counts = df_curves.groupby(["S0", "k", "n", "m", "m6"])["E*"].nunique()
        filtered_groups = group_counts[group_counts == self.CURVES_PER_MATERIAL].index
        filtered_df = df_curves[df_curves.set_index(["S0", "k", "n", "m", "m6"]).index.isin(filtered_groups)]
        sorted_df = filtered_df.sort_values(by=["S0", "k", "n", "m", "m6", "E*"])

        return sorted_df
    
