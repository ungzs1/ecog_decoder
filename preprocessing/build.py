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
    HtnetPreprocessor().__init__()
    HtnetPreprocessor().time_range = time_range
    HtnetPreprocessor().run()

