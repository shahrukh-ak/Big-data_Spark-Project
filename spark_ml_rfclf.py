from pyspark import SparkConf, SparkContext

from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext

from pyspark.ml.linalg import DenseVector

# Replace the line for LogisticRegression with RandomForestClassifier
# import RandomForestClassifier from pyspark.ml.classification
from pyspark.ml.classification import LogisticRegression

# Spark set-up
conf = SparkConf()
conf.setAppName("Random Forest Classifier")

sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")

spark = SparkSession(sc)

# Load dataset file as RDD
rdd = sc.textFile("/user/spark/iris.txt")
rdd = rdd.map(lambda x: x.split(','))

def renameLabel(x) :
	if x[4] == 'Iris-setosa':
		x[4] = 1
	elif x[4] == 'Iris-versicolor':
		x[4] = 2
	else:
		x[4] = 3
	return x

rdd = rdd.map(renameLabel)
rdd = rdd.map(lambda x: [float(x[0]), float(x[1]), float(x[2]), float(x[3]), int(x[4])]) 

# Create dataframe for ML model
df = spark.createDataFrame(rdd, ["sep_len", "sep_wid", "pet_len", "pet_wid", "class"])
data = df.rdd.map(lambda x: (DenseVector(x[:-1]), x[-1]))
df = spark.createDataFrame(data, ["features", "label"])

# Split data into train and test
train_data, test_data = df.randomSplit([.7,.3], seed=0)

# Declare ML model
# Change the LogisticRegression line below for RandomForestClassifier 
# RandomForestClassifier method accepts parameters like featuresCol, labelCol, and numTrees
# Run the file by varying number of trees and observe the performance 
logistic = LogisticRegression(featuresCol = "features", labelCol = "label", maxIter=10, regParam=0.1)

# Train the model using training data
# Change the line below according to the name of the algorithm instance variable
model = logistic.fit(train_data)

# Check the model on test data
predicted = model.transform(test_data)
predictAndLabel = predicted.select("prediction", "label")
print(predictAndLabel.show(20))

# There is no model summary for RF like Logistic Regression
# Use dataframe filter to find the number of records which are correctly predicted
# Divide the correctly predicted records with total records will give you accuracy
# Print the accuracy
