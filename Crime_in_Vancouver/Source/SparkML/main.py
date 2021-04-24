#!/usr/bin/env python3

#Spark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import *
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

## standard imports
import numpy as np
import pandas as pd
import random as rd
import seaborn as sns
import matplotlib.pyplot as plt

class RegressionModel(object):
    def __init__(self):
        self.spark_path = "/home/jim/spark-2.4.0-bin-hadoop2.7/jars/mysql-connector-java-5.1.49.jar"
        self.spark = SparkSession\
            .builder\
            .appName("Crime_Prediction")\
            .config("spark.driver.extraClassPath",self.spark_path)\
            .getOrCreate()
        self.spark_partition_id.sparkContext.setLogLevel('WARN')
        self.file_path = "/home/jim/Documents/Data_Analytics_Projects/Crime_in_Vancouver/Data/supercrime.csv"
        self.sc = self.spark.sparkContext
        self._run()

    def read_df(self):
        df = (self.spark.read.format("csv").option('header', 'true')\
              .load(self.file_path))
        # Caching this data frame as it's going to be read over and over again
        df = df.cache()
        return df
    
    def generate_localPlots(self,df):
        pdf = df.groupBy('NEIGHBOURHOOD').count().withColumnRenamed('count', 'CRIME_COUNT').toPandas()
        plt.rcParams["figure.figsize"] = [25, 10]
        sns.set(style="whitegrid")
        sns.set_color_codes("bright")

        bargraph = sns.barplot(x='CRIME_COUNT', y='NEIGHBOURHOOD', data=pdf)
        bargraph.set(ylabel="NEIGHBOURHOOD", xlabel="Crimes Record Count",)
        bargraph.savefig('Barplot.png')
        
        pdf_ = df.groupBy('YEAR').count().withColumnRenamed('count', 'CRIME_COUNT').toPandas()
        pdf_ = pdf_.astype({"YEAR": int})
        sns.set(style="whitegrid")
        sns.set_color_codes("bright")

        linegraph = sns.lineplot(x='YEAR', y='CRIME_COUNT', data=pdf_)
        linegraph.set(ylabel="YEAR", xlabel="Crimes Record Count")
        linegraph.savefig('Lineplot.png')
    
    def clean_df(self,df):
        tot = df.count()
        #Drop Nulls
        df = df.na.drop()
        print("Dropped {} rows with NULL values.".format(tot -df.count()))
        #Don't need street number for Hundred_BLOCK columns
        df = df.withColumn('HUNDRED_BLOCK', split(df['HUNDRED_BLOCK'], 'X ').getItem(1))
        #Type Cast columns
        df = df.select(col('TYPE'),
               col('YEAR').cast('int'),
               col('MONTH').cast('int'),
               col('DAY').cast('int'),
               col('HUNDRED_BLOCK'),
               col('NEIGHBOURHOOD'),
               col('HOUR').cast('int'),
               col('MINUTE').cast('int'),
               col('LATITUDE').cast('float'),
               col('LONGITUDE').cast('float'),
               col('FACILITY'),
               col('STATION'),
               col('SCHOOL')               
                        )
        return df
        
    def extract_features(self,df):
        features = [i for i in df.columns if i != "NEIGHBOURHOOD"]
        return features
    
    def index_df(self,df):
        features = self.extract_features(df)
        for f in features:
            if dict(df.dtypes)[f] =='string':
                df = StringIndexer(inputCol=f,outputCol=f+'_indexed').setHandleInvalid("keep").fit(df).transform(df)
                df = df.drop(f)
        #Feature vector column with all the features smashed together     
        new_features = self.extract_features(df)
        assembler = VectorAssembler(inputCols=new_features, outputCol='Features')
        self.vectorized_df = assembler.transform(df)
        self.vectorized_df = self.vectorized_df.drop(*features)
        #Indexing label column
        indexed_df = StringIndexer(inputCol='NEIGHBOURHOOD', outputCol='Neighbourhood').setHandleInvalid("keep").fit(self.vectorized_df).transform(self.vectorized_df)
        #Fixed Bug
        sqlContext.sql("set spark.sql.caseSensitive=true")
        indexed_df = indexed_df.drop('NEIGHBOURHOOD')
        return indexed_df
    
    def assign_classWeights(self,df):
        BalancingRatio = df.groupby('Neighbourhood').count().select('count')\
                                    .rdd.max().asDict()['count']/df.count()
        weighted_df = df.withColumn("classWeights", \
                    when(df.Neighbourhood == 0.0,1-BalancingRatio).otherwise(BalancingRatio))
        return weighted_df
    
    def logistic_regression(self,indexed_df):
        (training_data, test_data) = indexed_df.randomSplit([0.8,0.2])
        weighted_training_data = self.assign_classWeights(training_data)
        logisticRegression = LogisticRegression(labelCol='Neighbourhood',
                                        featuresCol='Features',
                                        maxIter=10,
                                        family='multinomial').setWeightCol("classWeights")
        self.model = logisticRegression.fit(weighted_training_data)
        print("The Accuracy of the model is {} %".format(self.model.summary.accuracy*100))
        print("The co-efficient matrix of the model is as follows:\n{}".format(self.model.coefficientMatrix))
        plot = sns.barplot(y=self.model.summary.objectiveHistory, x=list(range(len(self.model.summary.objectiveHistory))))
        plot.savefig('Model_ObjectiveHistory.png')
        self.model.save(self.sc, "logisticRegression.model")
        
    def predict(self):
        df = self.df
        f = self.extract_features(df)
        df = df.drop('NEIGHBOURHOOD','YEAR')
        df = self.index_df(df)
        vectorized_df = VectorAssembler(inputCols=features, outputCol='Features').transform(df)
        vectorized_df = vectorized_df.drop(*f)
        pdf = df.toPandas()
        df['YEAR'] = np.random.randint(2019,2026, size=len(pdf))
        df = self.spark.createDataFrame(pdf)
        local_model = self.model
        predictions = local_model.transform(self.vectorized_df)
        #Save predictions dataframe locally
        predictions.repartition(1).write.format("com.databricks.spark.csv").option("header", "true").save("Predictions.csv")
    
    def _run(self):
        self.df = self.read_df()
        self.generate_localPlots(self.df)
        self.df = self.clean_df(self.df)
        self.ixdf = self.index_df(self.df)
        self.logistic_regression(self.ixdf)
        
if __name__ == '__main__':
    r = RegressionModel()
    r.predict()
    
    
    
    
    
    
    
    
    
    
    