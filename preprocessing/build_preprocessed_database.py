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

### UNCOMMENT THE PROPER LINE TO BULD THE PREPROCESSED DATABASE

#StanfordPreprocessor().run()
#Bci3Preprocessor().run()
HTNet_preprocessor().run()

