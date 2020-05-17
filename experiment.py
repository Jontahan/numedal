class Experiment:
    def __init__(self, environment_params={}, training_params={}):
        self.environment_params = environment_params
        self.training_params = training_params
        self.file_prefix = ''