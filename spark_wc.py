#Command: spark-submit spark_wc.py
from pyspark import SparkConf, SparkContext

#Spark set-up
conf = SparkConf()
conf.setAppName("Word count App")
sc   = SparkContext(conf=conf)

#uncomment the sc.setLoglevel line, when your program works fine. 
#Run the program again to take the screenshot.
#sc.setLogLevel("WARN")

# Upload data file in Hadoop and provide its path in textFile function
rdd  = sc.textFile("/user/spark/words.txt")
rdd  = rdd.flatMap(lambda x: x.split(' '))
rdd	 = rdd.map(lambda x: (x, 1))
# Add few lines of code below

rdd  = rdd.reduceByKey(lambda x,y: x+y)
# Add your code below
#
#
# you may store top 10 results in 'out' variable
# and use it to display as mentioned below.
for item in out:
	print item[0],'\t:\t',str(item[1])

