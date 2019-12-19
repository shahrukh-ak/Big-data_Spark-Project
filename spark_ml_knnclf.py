from pyspark import SparkConf, SparkContext
from pyspark.sql import Row, SparkSession, SQLContext
from pyspark.sql import functions
from pyspark.sql.types import FloatType, IntegerType, BooleanType

import math

def renameLabel(x) :
	if x[4] == 'Iris-setosa':
		x[4] = 1
	elif x[4] == 'Iris-versicolor':
		x[4] = 2
	else:
		x[4] = 3
	return x

# [2 points] Define compute_dis function that takes eight float type parameters 
# that hold attributes for two Iris flower records. The function will compute the distance 
# between the records and return it.

def compute_dis(x1, x2, x3, x4, y1, y2, y3, y4):
	# your code
	return root_val

# [3 points] Define get_actual_class that will take a list as parameter and return an integer 
# if all the elements in the list are of that integer
def get_actual_class(actual_class_list):
	# your code
	return # something

# [4 points] Define get_predicted_class that will take k, a list of distances, 
# and a list of the classes of the training data for which the distances have been calculated. 
# The function will sort the distances list and training classes list. And returns the majority 
# class for k shortest distances points.

def get_predicted_class(k, dis_list, trn_class):
	pred_class = None 
	# your code
	return pred_class


# Spark set-up
conf = SparkConf()
conf.setAppName("K Nearest Neighbors")

sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")

spark = SparkSession(sc)
sqlContext = SQLContext(sc)

# [2 points] Register the functions that you defined earlier as user-defined functions 
# for using with SQL queries. One example is given below
sqlContext.udf.register("compute_dis", compute_dis, FloatType())
# Your code

# Load dataset file as RDD
rdd = sc.textFile("/user/spark/iris.txt")
rdd = rdd.map(lambda x: x.split(','))

rdd = rdd.map(renameLabel)
rdd = rdd.map(lambda x: [float(x[0]), float(x[1]), float(x[2]), float(x[3]), int(x[4])]) 

# Create dataframe
df = spark.createDataFrame(rdd, ["sep_len", "sep_wid", "pet_len", "pet_wid", "class"])
train_data, test_data = df.randomSplit([.7, .3], seed=0)

# [2 points] Register train_data and test_data as iris_train and iris_test tables 
# using registerTempTable
# Your code 


# [5 points] Write SQL using sqlContext.sql() function that returns sepal length, 
# sepal width, petal length, and petal width for each iris_test record along with its 
# distance from all the iris_train records, class of iris_test and class of iris_train 
# You need to perform cross join of iris_test with iris_train tables
# Refer https://www.w3resource.com/sql/joins/cross-join.php for cross join

# Your SQL query

# [5 points] For each iris_test record, you will see many rows that represent its distances from 
# the records in iris_train. Now, you need to group the distances, iris_test class, 
# iris_train class for each specific iris_test record using groupBy and agg functions on 
# dataframe returned from previous query. Following will be helpful for this task:
# https://danvatterott.com/blog/2018/09/06/python-aggregate-udfs-in-pyspark/
# get_actual_class function that you implemented earlier

# Your command for dataframe groupby

# [3 points] Now convert the iris_test class list in each record of iris_test to one class using 
# get_actual_class function that you implemented earlier
# Your code

# [2 points] You may save the previous results in a dataframe, use that dataframe to get the 
# predicted class for each iris_test record using get_predicetd_class function
# Your code

# [2 points] Now find the accuracy and print it.

