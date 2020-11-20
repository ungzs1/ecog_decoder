try:
    from .preprocessor_base import *
except ImportError:
    from preprocessor_base import *

class StanfordPreprocessor(Preprocessor):

	def __init__(self, *args, **kwargs):
		super(StanfordPreprocessor, self).__init__(*args, **kwargs)
		self.config.["data_source"] = "stanford"

	def load_data_and_labels(self, filename):
        # should return a pair of numpy arrays of dimensions ( [timestep, channels], [timestep, label] )
        x = []
        y = []

        return x, y


    def train_files_from_dir(self):
        # return all the valid train files in a list
        file_list = []

        return file_list
    
    def test_files_from_dir(self):
        # return all the valid test files in a list
        file_list = []

        return file_list 