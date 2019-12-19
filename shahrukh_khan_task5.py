#import libraries
from pyspark import SparkConf, SparkContext

from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext

from pyspark.ml.linalg import DenseVector
from pyspark.ml.classification import LogisticRegression

#Spark set-up
conf = SparkConf()
conf.setAppName("Logistic regression")

sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")

spark = SparkSession(sc)

#Load dataset file as RDD
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

#Create dataframe for ML model
df = spark.createDataFrame(rdd, ["sep_len", "sep_wid", "pet_len", "pet_wid", "class"])
data = df.rdd.map(lambda x: (DenseVector(x[:-1]), x[-1]))
df = spark.createDataFrame(data, ["features", "label"])

#Split data into train and test
train_data, test_data = df.randomSplit([.7,.3], seed=0)

#Declare ML model
logistic = LogisticRegression(featuresCol = "features", labelCol = "label", maxIter=20, regParam=0.5)

# Train the model using training data
model = logistic.fit(train_data)

#Check the model on test data
predicted = model.transform(test_data)
predictAndLabel = predicted.select("prediction", "label")
print(predictAndLabel.show(20))

#Model stats
accuracy = model.summary.accuracy
precision = model.summary.weightedPrecision
recall = model.summary.weightedRecall
fMeasure = model.summary.weightedFMeasure()
print("Accuracy: %0.2f\nPrecision: %0.3f\nRecall: %0.3f\nF-measure: %0.3f"
      % (accuracy, precision, recall, fMeasure))
