# Big-data_Spark-hands-on
- Task 2: Write Spark application that reads uploaded words.txt file from HDFS cluster and finds top 10 most frequent words and their frequencies. In the text file, a few words may appear in different forms, e.g. The, the, you have to treat them same. In addition, some words may have double quote, single quote or other non-alphabet character in the prefix or suffix, your program should be able to remove them and then consider the remaining characters as word. <br>
- Task 3: Write Spark application that reads uploaded sales.txt file from HDFS cluster and finds average and standard deviation of stores’ sales in each city. <br>
- Task 4: Upload airfoil.txt file in HDFS cluster and run spark_ml_linreg.py Spark application. The application is Linear Regression implementation in Spark. The dataset used in this task is iris.names. Run the application by varying parameters like maxIter, regParam, and elasticNetParam. Mention your observation in the report. You may refer the following link:
https://www.datacamp.com/community/tutorials/apache-spark-tutorial-machine-learning <br>
- Task 5: Upload iris.txt file in HDFS cluster and run spark_ml_lrclf.py Spark application. The application is Logistic Regression implementation in Spark. The dataset used in this task is taken from iris.names.<br>
- Task 6: Use the uploaded iris.txt dataset in HDFS cluster and build a Random Forest classifier model. You can use spark_ml_rfclf.py for this task. Compare your results with Logistic regression implementation.<br>
- Task 7: Implement K-Nearest Neighbors from scratch using the skeleton code given in spark_ml_knnclf.py. The implementation is discussed using Spark dataframe and Spark SQL features and designed for iris.txt dataset only

