try:
    from .preprocessor_stanford_motor import *
except ImportError:
    from preprocessor_stanford_motor import *

try:
    from .preprocessor_bci_iii import *
except ImportError:
    from preprocessor_bci_iii import *

try:
    from .preprocessor_HTNet_data import *
except ImportError:
    from preprocessor_HTNet_data import *

# ** USE THIS PYTHON FILE TO EASILY BUILD PREPROCESSED DATASETS WITH DIFFERENT PARAMETERS **
#StanfordPreprocessor().run()
#Bci3Preprocessor().run()

time_ranges = [(500, -1), (500, -100), (500, -200), (500, -300), (500, -400)]
for time_range in time_ranges:
    preprocessor = HtnetPreprocessor()
    preprocessor.time_range = time_range
    preprocessor.config['save_name'] = "HTNet_data_{}_{}.hdf5".format(time_range[0],
                                                                      time_range[1])
    preprocessor.config['default_config_name'] = "HTNet_CONFIG_{}_{}".format(time_range[0],
                                                                             time_range[1])
    print(preprocessor.config['save_name'])
    preprocessor.run()

