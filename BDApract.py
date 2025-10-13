Practical 2: Implement a program in pig
rom pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg

spark = SparkSession.builder.appName('HelloWorld').getOrCreate()

data = spark.read.csv('/content/Implement_pig_using_pyspark/Real estate (1).csv', 
                      inferSchema=True, header=True)
data.printSchema()

data.show()

filter_data = data.filter(col('X2 house age') > 20)
filter_data.show()

grouped_data = data.groupBy('X4 number of convenience stores').agg(
    avg('Y house price of unit area')).alias('avg house price')
grouped_data.show()

spark.stop()
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Practical 3: Implement Word count/ frequency program using MapReduce
from pyspark.sql import SparkSession
from pyspark import SparkContext

spark = SparkSession.builder.master('local[*]').getOrCreate()
sc = SparkContext.getOrCreate()

words = sc.textFile('/content/Implement-word-count-frequency-programs-using-MapReduce/input.txt')

words.collect()

Word_count = words.flatMap(lambda line: line.split(' ')).map(lambda word: (word, 1))
Word_count.count()

Word_count.collect()

distintwordcount = Word_count.reduceByKey(lambda a, b: a+b)
distintwordcount.count()

distintwordcount.collect()

sortedwordcount = distintwordcount.map(lambda a: (a[1], a[0])).sortByKey()
sortedwordcount.collect()

sortedwordcount.top(20)
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Practical 4: configure hive and implement the application in hive 
from pyspark.sql import SparkSession, Row

spark = SparkSession.builder.appName('Hive').enableHiveSupport().getOrCreate()

data = [
    Row(name='Farhaan', Age=22, Dept='DA'),
    Row(name='KamalKishor', Age=22, Dept='DA'),
    Row(name='Vaibhav', Age=23, Dept='DE'),
    Row(name='Hussain', Age=23, Dept='DS'),
    Row(name='Kotesh', Age=22, Dept='DS'),
    Row(name='Isaqe', Age=28, Dept='CE'),
    Row(name='Shadab', Age=22, Dept='DA')
]

df = spark.createDataFrame(data)
df.show()

spark.sql('CREATE DATABASE IF NOT EXISTs EmployeeData_db')

spark.sql('USE EmployeeData_db')

df.write.mode('overwrite').saveAsTable('EmployeeData_db.EmployeeData')

result = spark.sql('SELECT * FROM EmployeeData')
result.show()

spark.stop()
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Practical 5: Implement Spark SQL
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('House Price').getOrCreate()

df = spark.read.csv('/content/sample_data/california_housing_train.csv', inferSchema=True, header=True)
df.show()

df.createOrReplaceTempView('Housing')

spark.sql('SELECT * FROM Housing').show()

spark.sql('''SELECT total_rooms, total_bedrooms, population 
             FROM Housing
             ORDER BY population DESC
             LIMIT 10
''').show()

spark.sql('''SELECT 
             latitude, longitude, SUM(total_rooms) AS total_rooms
             FROM Housing
             GROUP BY latitude, longitude''').show()

spark.stop()
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Practical 6: Implement Machine Learning using Pyspark
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('LinearRegression').getOrCreate()

data = spark.read.csv('/content/Pyspark_Linear_Regression/Real estate.csv', 
                      inferSchema=True, header=True)

data.printSchema()

data.show()

data.describe().show()

assembler = VectorAssembler(
    inputCols=['X1 transaction date',
               'X2 house age',
               'X3 distance to the nearest MRT station',
               'X4 number of convenience stores',
               'X5 latitude',
               'X6 longitude',
               ],
    outputCol='features'
)

dataset = assembler.transform(data)
dataset.show()

dataset.select(['features', 'Y house price of unit area']).show()

train_data, test_data = dataset.randomSplit([0.7, 0.3])

train_data.show(truncate=False)

test_data.show()

lr = LinearRegression(featuresCol='features', labelCol='Y house price of unit area')
lrmodel = lr.fit(train_data)

test_stats = lrmodel.evaluate(test_data)

print('MSE:', test_stats.meanSquaredError)
print('R2:', test_stats.r2)
print('RMSE:', test_stats.rootMeanSquaredError)
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Practical 7: Implement Spark Streaming
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
import threading as th
import time as t
import socket as s
from sklearn.datasets import fetch_20newsgroups

sc = SparkContext("local[2]", "InbuiltDataSetStreaming") #  <-- Create a SparkContext running locally with 2 threads; name the app "InbuiltDatasetStreaming"
#     "local[2]" means Spark will use 2 worker threads (one can handle receiving, one processing).
ssc = StreamingContext(sc, 3) # <-- Create StreamingContext with batch duration 3 seconds.
#     This means Spark will collect data for 3s and then process it as one micro-batch.

lines = ssc.socketTextStream('localhost', 9999)

words = lines.flatMap(lambda line: line.split(" "))
pairs = words.map(lambda word: (word, 1))
word_counts = pairs.reduceByKey(lambda x, y: x + y)

word_counts.pprint()

word_counts.saveAsTextFiles("stream_output/wordcount")

def start_streaming():
  ssc.start()
  ssc.awaitTermination(60)
  ssc.stop(stopSparkContext=True, stopGraceFully=True)

def send_data():
  dataset = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
  data = dataset.data[:50]

  server = s.socket(s.AF_INET, s.SOCK_STREAM)
  server.bind(('localhost', 9999))
  server.listen(1)
  print('Socket server started on port 9999....')
  conn, addr = server.accept()

  for doc in data:
    line = doc.split('\n')[0]
    if line.strip():
      conn.send((line + '\n').encode('utf-8'))
      time.sleep(2)

  conn.close()
  server.close()

th.Thread(target=send_data).start()
start_streaming()
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Practical 8: Demonstrate Spark shell commends using logistic regression
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import isnull, when, count, col
from pandas.plotting import scatter_matrix
import pandas as pd 

spark = SparkSession.builder.appName('ml-diabetes').getOrCreate()

df = spark.read.csv('/content/Logistic_regression_BDA_Classification_Pyspark/diabetes.csv', header = True, inferSchema = True)
df.printSchema()

pd.DataFrame(df.take(5), columns=df.columns).transpose()

df.show()

df.toPandas()

df.groupby('Outcome').count().toPandas()

numeric_features = [t[0] for t in df.dtypes if t[1] == 'int']

numeric_features

df.select(numeric_features).describe().toPandas().transpose()

numeric_data = df.select(numeric_features).toPandas()

axs = scatter_matrix(numeric_data, figsize=(8, 8));

# Rotate axis labels and remove axis ticks
n = len(numeric_data.columns)
for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(0)
    v.yaxis.label.set_ha('right')
    v.set_yticks(())
    h = axs[n-1, i]
    h.xaxis.label.set_rotation(90)
    h.set_xticks(())

df.select([count(when(isnull(c), c)).alias(c) for c in df.columns]).show()

dataset = df.drop('SkinThickness')
dataset = dataset.drop('Insulin')
dataset_new = dataset.drop('DiabetesPedigreeFunction')
dataset_final = dataset_new.drop('Pregnancies')

dataset_final.show()

required_features = ['Glucose', 'BloodPressure', 'BMI', 'Age']

assembler = VectorAssembler(inputCols=required_features, outputCol='features')

transformed_data = assembler.transform(dataset_final)
transformed_data.show()

(training_data, test_data) = transformed_data.randomSplit([0.8,0.2], seed =2020)

print("Training Dataset Count: " + str(training_data.count()))
print("Test Dataset Count: " + str(test_data.count()))

lr = LogisticRegression(featuresCol = 'features', labelCol = 'Outcome', maxIter=10)
lrModel = lr.fit(training_data)
lr_predictions = lrModel.transform(test_data)

multi_evaluator = MulticlassClassificationEvaluator(labelCol = 'Outcome', metricName = 'accuracy')
print('Logistic Regression Accuracy:', multi_evaluator.evaluate(lr_predictions))
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Practical 9: Implement Decision tree classification technique
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import isnull, when, count, col
from pandas.plotting import scatter_matrix
import pandas as pd 
spark = SparkSession.builder.appName('ml-diabetes').getOrCreate()

df = spark.read.csv('/content/Logistic_regression_BDA_Classification_Pyspark/diabetes.csv', header = True, inferSchema = True)
df.printSchema()

pd.DataFrame(df.take(5), columns=df.columns).transpose()

df.show()

df.toPandas()

df.groupby('Outcome').count().toPandas()

numeric_features = [t[0] for t in df.dtypes if t[1] == 'int']

numeric_features

df.select(numeric_features).describe().toPandas().transpose()

numeric_data = df.select(numeric_features).toPandas()

axs = scatter_matrix(numeric_data, figsize=(8, 8));

# Rotate axis labels and remove axis ticks
n = len(numeric_data.columns)
for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(0)
    v.yaxis.label.set_ha('right')
    v.set_yticks(())
    h = axs[n-1, i]

df.select([count(when(isnull(c), c)).alias(c) for c in df.columns]).show()

dataset = df.drop('SkinThickness')
dataset = dataset.drop('Insulin')
dataset_new = dataset.drop('DiabetesPedigreeFunction')
dataset_final = dataset_new.drop('Pregnancies')

dataset_final.show()

required_features = ['Glucose', 'BloodPressure', 'BMI', 'Age']

assembler = VectorAssembler(inputCols=required_features, outputCol='features')

transformed_data = assembler.transform(dataset_final)
transformed_data.show()

(training_data, test_data) = transformed_data.randomSplit([0.8,0.2], seed =2020)

print("Training Dataset Count: " + str(training_data.count()))
print("Test Dataset Count: " + str(test_data.count()))

#Decision Tree Classifier
dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'Outcome', maxDepth = 3)
dtModel = dt.fit(training_data)

dt_predictions = dtModel.transform(test_data)
dt_predictions.select('Glucose', 'BloodPressure', 'BMI', 'Age', 'Outcome').show(10)

multi_evaluator = MulticlassClassificationEvaluator(labelCol = 'Outcome', metricName = 'accuracy')
print('Decision Tree Accuracy:', multi_evaluator.evaluate(dt_predictions))

#Gradient-boosted Tree classifier Model
gb = GBTClassifier(labelCol = 'Outcome', featuresCol = 'features')
gbModel = gb.fit(training_data)
gb_predictions = gbModel.transform(test_data)

multi_evaluator = MulticlassClassificationEvaluator(labelCol = 'Outcome', metricName = 'accuracy')
print('Gradient-boosted Trees Accuracy:', multi_evaluator.evaluate(gb_predictions))
