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
	root_val = ((x1 - y1)**2 + (x2 - y2)**2 + (x3 - y3)**2 + (x4 - y4)**2)**0.5
	return root_val

# [3 points] Define get_actual_class that will take a list as parameter and return an integer 
# if all the elements in the list are of that integer
def get_actual_class(actual_class_list):
	t = set(actual_class_list)
	if len(t) == 1:
		return t.pop()
	return

# [4 points] Define get_predicted_class that will take k, a list of distances, 
# and a list of the classes of the training data for which the distances have been calculated. 
# The function will sort the distances list and training classes list. And returns the majority 
# class for k shortest distances points.

def get_predicted_class(k, dis_list, trn_class):
	def first(n): return n[0]

	t = list(zip(dis_list, trn_class))
	t.sort(key=first)
	t = t[:k]
	t = [x[1] for x in t]

	pred_class = [(t.count(i), i) for i in (1,2,3)]
	pred_class.sort(key=first, reverse=True)
	
	return pred_class[0][1]


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
sqlContext.udf.register("get_actual_class", get_actual_class, IntegerType())
sqlContext.udf.register("get_predicted_class", get_predicted_class, IntegerType())
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
train_data.registerTempTable('iris_train')
test_data.registerTempTable('iris_test')



# [5 points] Write SQL using sqlContext.sql() function that returns sepal length, 
# sepal width, petal length, and petal width for each iris_test record along with its 
# distance from all the iris_train records, class of iris_test and class of iris_train 
# You need to perform cross join of iris_test with iris_train tables
# Refer https://www.w3resource.com/sql/joins/cross-join.php for cross join

t = sqlContext.sql("SELECT iris_test.sep_len test_sep_len, iris_test.sep_wid test_sep_wid, iris_test.pet_len test_pet_len, \
						iris_test.pet_wid test_pet_wid, iris_test.class test_class, iris_train.sep_len train_sep_len, \
						iris_train.sep_wid train_sep_wid, iris_train.pet_len train_pet_len, iris_train.pet_wid train_pet_wid, \
						iris_train.class train_class FROM iris_test CROSS JOIN iris_train")

t.registerTempTable('crossiris')
t = sqlContext.sql("SELECT test_sep_len, test_sep_wid, test_pet_len, test_pet_wid, test_class, compute_dis(test_sep_len, test_sep_wid, test_pet_len, \
						test_pet_wid, train_sep_len, train_sep_wid, train_pet_len, train_pet_wid) distance, train_class FROM crossiris")

# [5 points] For each iris_test record, you will see many rows that represent its distances from 
# the records in iris_train. Now, you need to group the distances, iris_test class, 
# iris_train class for each specific iris_test record using groupBy and agg functions on 
# dataframe returned from previous query. Following will be helpful for this task:
# https://danvatterott.com/blog/2018/09/06/python-aggregate-udfs-in-pyspark/
# get_actual_class function that you implemented earlier

t = t.groupBy('test_sep_len', 'test_sep_wid', 'test_pet_len', 'test_pet_wid').agg(functions.collect_list('test_class').alias('test_class'), \
						functions.collect_list('train_class').alias('train_class'), functions.collect_list('distance').alias('distance'))

t.registerTempTable('table2')

# Your command for dataframe groupby

# [3 points] Now convert the iris_test class list in each record of iris_test to one class using 
# get_actual_class function that you implemented earlier
# Your code
t = sqlContext.sql("SELECT test_sep_len, test_sep_wid, test_pet_len, test_pet_wid, get_actual_class(test_class) actual_class, \
						train_class, distance FROM table2")

t.registerTempTable('table3')

# [2 points] You may save the previous results in a dataframe, use that dataframe to get the 
# predicted class for each iris_test record using get_predicetd_class function
# Your code

t = sqlContext.sql("SELECT test_sep_len, test_sep_wid, test_pet_len, test_pet_wid,actual_class, \
						get_predicted_class(30, distance, train_class) predicted_class FROM table3")
t.show()

# [2 points] Now find the accuracy and print it.
trues = t.filter('actual_class == predicted_class').count()
total = t.count()

print("Accuracy: %0.2f" % (float(trues)/float(total)))

