import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense
from tqdm._tqdm_notebook import tqdm_notebook
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TFBertModel
tqdm_notebook.pandas()