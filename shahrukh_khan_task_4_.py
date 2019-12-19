#import libraries
from pyspark import SparkConf, SparkContext

from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext

from pyspark.ml.linalg import DenseVector
from pyspark.ml.feature import StandardScaler
from pyspark.ml.regression import LinearRegression

#Spark set-up
conf = SparkConf()
conf.setAppName("Linear Regression")

sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")

spark = SparkSession(sc)

#Load dataset file as RDD
rdd = sc.textFile("/user/spark/airfoil.txt")
rdd = rdd.map(lambda x: x.split('\t'))
rdd = rdd.map(lambda x: [float(x[0]), float(x[1]), float(x[2]), float(x[3]),
 float(x[4]), float(x[5])]) 

#Create dataframe for ML model
df = spark.createDataFrame(rdd, ["frequency", "angle", "chord", "velocity", "suction", "pressure"])
data = df.rdd.map(lambda x: (DenseVector(x[:-1]), x[-1]))
df = spark.createDataFrame(data, ["features", "label"])

#Feature scaling
standardScaler = StandardScaler(inputCol="features", outputCol="features_scaled")
scaler = standardScaler.fit(df)
scaled_df = scaler.transform(df)

#Split data into training and test
train_data, test_data = scaled_df.randomSplit([.7,.3],seed=1234)
train_data = train_data.select("features_scaled", "label")
test_data = test_data.select("features_scaled", "label")
train_data = train_data.withColumnRenamed("features_scaled", "features")
test_data = test_data.withColumnRenamed("features_scaled", "features")

#Declare regression ML model
lr = LinearRegression(labelCol="label", maxIter=20, regParam=0.6, elasticNetParam=0.16)

#Train model on training data
linearModel = lr.fit(train_data)

#Test model on test data
predicted = linearModel.transform(test_data)
predictions = predicted.select("prediction").rdd.map(lambda x: x[0])
labels = predicted.select("label").rdd.map(lambda x: x[0])
predictionAndLabel = predictions.zip(labels).collect()
print(predictionAndLabel[:5])

#model stats
#linearModel.coefficients
#linearModel.intercept
print('RMSE: '+str(linearModel.summary.rootMeanSquaredError))
print('R2: '+str(linearModel.summary.r2))
