try:
    from .preprocessor_stanford_motor import *
except ImportError:
    from preprocessor_stanford_motor import *

StanfordPreprocessor().run()
