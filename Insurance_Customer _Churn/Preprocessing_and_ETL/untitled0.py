import pandas as pd
from pyspark.sql import SparkSession, functions, types
from pyspark import SparkConf, SparkContext
spark = SparkSession.builder.appName('example code').getOrCreate()