import tensorflow as tf
import numpy as np
import argparse
import nltk



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Unit testing')
    parser.add_argument('function_name')
    args = parser.parse_args()
