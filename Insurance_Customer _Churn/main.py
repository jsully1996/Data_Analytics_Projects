from flask import Flask, request, redirect, render_template, url_for
#Import dependencies
from pyspark.sql import SparkSession, functions, types
from pyspark.ml.feature import StringIndexer
from pysparkling import *
import h2o
app = Flask(__name__)

class prediction_model(object):
    def __init__(self,customer):        
        #Import Pyspark and create a SparkSession, set the Sparkcontext.
        self.spark = SparkSession\
            .builder\
            .appName("Front-end")\
            #.config("spark.driver.extraClassPath","/home/jim/spark-2.4.0-bin-hadoop2.7/jars/mysql-connector-java-5.1.49.jar")\
            .getOrCreate()
        self.spark.sparkContext.setLogLevel('WARN')
        self.sc = self.spark.sparkContext
        #Create an H2O cluster inside the Spark cluster
        self.hc = H2OContext.getOrCreate(self.spark)
        self.customer=customer
        
    def standardize(self,c):
        c['Customer Lifetime Value'] = float(c['Customer Lifetime Value'])
        c['Age'] = int(c['Age'])
        c['Income'] = int(c['Income'])
        c['Monthly Premium Auto'] = int(c['Monthly Premium Auto'])
        c['Total Written Premium'] = int(c['Total Written Premium'])
        c['Losses'] = int(c['Losses'])
        c['Loss Ratio'] = float(c['Loss Ratio'])
        c['Growth Rate'] = float(c['Growth Rate'])
        c['Commissions'] = int(c['Commissions'])
        c['Months Since Last Claim'] = int(c['Months Since Last Claim'])
        c['Months Since Policy Inception'] = int(c['Months Since Policy Inception'])
        c['Number of Open Complaints'] = int(c['Number of Open Complaints'])
        c['Number of Policies'] = int(c['Number of Policies'])
        c['Number of previous policies'] = int(c['Number of previous policies'])
        c['Total Claim Amount'] = float(c['Total Claim Amount'])
        return c
        
    def transform_data(self,df):
        #Transform it to a format that will be accepted by the model
            cols_to_drop = ["Customer ID","Name","Address","Phone_no","Email","SSN"]
            df = df.drop(*cols_to_drop)
            df = df.withColumn("Customer Lifetime Value", functions.round("Customer Lifetime Value", 2))
            df = df.withColumn("Loss Ratio", functions.round("Loss Ratio", 3))
            df = df.withColumn("Growth Rate", functions.round("Growth Rate", 3))
            df = df.withColumn("Total Claim Amount", functions.round("Total Claim Amount", 3))
            df = df.withColumn("Job",functions.split("Job", ",").getItem(0))
            df = df.withColumn("Company",functions.reverse(functions.split("Company", ",")).getItem(0))
            indexer_list = []
            categ_cols = ['City','Response','Coverage','Education','Employment_Status','Gender','Location_Code','Marital Status','Policy_Type','Policy_Rating','Renew_Offer_Type','Sales_Channel','Total Claim Amount','Feedback','Job','Company','Credit Card Provider']
            for i in categ_cols:
                if i == 'City':
                    indexer_list.append(StringIndexer(inputCol=i, outputCol=i+"Index"))
                else:
                    indexer_list.append(StringIndexer(inputCol=i, outputCol=i+" Index"))
            for j in indexer_list:
                df = j.fit(df).transform(df)
            df = df.select([c for c in df.columns if c not in categ_cols])
            df = df.withColumn("Effective To Date",functions.split("Effective To Date", "-").getItem(2))
            df = df.withColumn("Effective To Date", df["Effective To Date"].cast(types.IntegerType()))
            return df
    
    def predict(self):
            #Resuse the Saved Random Forest Model
            model = h2o.load_model('Misc/models/RF_Insurance_model/DRF_model_python_1591600273347_1')
            #Convert it to a Spark Dataframe
            customer = self.standardize(self.customer)
            org_df = self.sc.parallelize([customer]).toDF()
            df = self.transform_data(org_df)
            #Obtain a result using the saved model
            pred = model.predict(self.hc.asH2OFrame(df))
            prediction_df = self.hc.asSparkFrame(pred)
            prediction_df = prediction_df.withColumn("predict", functions.round("predict", 0))
            prediction_df = prediction_df.withColumn("predict", prediction_df["predict"].cast(types.IntegerType()))
            result = prediction_df.collect()[0].predict   
            return result
    
    def get_answer(self,result):
        if result is 0:
            return 'This customer is not likely to Churn'
        elif result is 1:
            return 'This Customer is likely to Churn'
    
    def run(self):
        self.prediction = self.predict()
        self.answer = self.get_answer(self.prediction)
        return self.answer

#Displays result of prediction by the model
@app.route('/<nm>')
def name(nm):
    return f"<h1>{nm}</h1>"

@app.route('/', methods=['GET', 'POST'])
def my_form():
    if request.method == "POST":
        name = request.form
        #print(type(name['Age']))
        p = prediction_model(dict(name))
        ans = p.run()
        return redirect(url_for('name',nm=ans))
    else:
        return render_template('form.html')

    
if __name__ == "__main__":
    app.run(debug=True)
    
