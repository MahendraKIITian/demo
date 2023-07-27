import pandas as pd
#import json
import datetime
class ModelVersioning:
    def __init__(self, trained_model):
        self.trained_model = trained_model
        #self.model_name = model_name

    def Model_HyperParameters(self):
        df_param = pd.DataFrame([self.trained_model.get_params()]).T.reset_index().rename(
            columns={'index': 'Hyper_Parameter_name', 0: 'Hyper_Parameter_value'})
        return df_param

    def Model_Artifacts(self):
        df_params = self.Model_HyperParameters()
        df_params['run_time'] = datetime.datetime.now()
        df_params['model_name'] = "ML Model"
        df_params['model_type'] = "Classification"
        return df_params








