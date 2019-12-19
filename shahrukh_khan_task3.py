# Command: spark-submit spark_std.py
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql import SQLContext

# Spark set-up
conf = SparkConf()
conf.setAppName("Purchase App")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

# uncomment the sc.setLoglevel line, when your program works fine. 
# Run the program again to take the screenshot.
#sc.setLogLevel("WARN")

# Upload data file in Hadoop and provide its path in textFile function
rdd = sc.textFile("/user/spark/sales.txt")

# Add a few lines of code here to split 
# the attributes in each line, pick only required attributes,
# cast attributes type if needed.
rdd_s = rdd.map(lambda x:x.split('\t'))
rdd1 = rdd_s.map(lambda x:(x[2],float(x[4])))

# Add code to convert RDD to dataframe


# create SQL table from data frame.


sqlContext = SQLContext(sc)
data = sqlContext.createDataFrame(rdd1)
data.createOrReplaceTempView('tmp_table')
out = sqlContext.sql('select _1 , avg(_2),std(_2) from tmp_table group by _1')
# Write query using sqlContext.sql() function

# You may convert SQL dataframe in RDD
# and use it for pretty formatting as mentioned below
# city\t(average sale with 2 digits after decimal)\t(standard deviation in sale with 3 digits after decimal) 
# For example:
# Las Vegas	1200.56	23.321
rdd2= out.rdd.map(list)
rdd3 =rdd2.map(lambda x:(x[0],x[1],x[2]))

for i in rdd3.collect():
    print(i)
