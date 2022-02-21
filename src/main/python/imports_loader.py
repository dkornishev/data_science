from bs4 import BeautifulSoup
import nltk

from requests import get
from nltk import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
import collections
import pandas as pd
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import copy
import statsmodels.api as sm
import tensorflow