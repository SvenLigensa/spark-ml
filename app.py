import os
import logging
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize logging
#logging.basicConfig(
#    level=logging.DEBUG,
#    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#    handlers=[logging.FileHandler("app.log")]
#)
#logger = logging.getLogger("Spark Application")

# Initialize Spark
conf = SparkConf()
conf.setAppName("Spark Application")
conf.setMaster("local")
conf.set("spark.hadoop.fs.defaultFS", "file:///")
sc = SparkContext.getOrCreate(conf)
sc.setLogLevel("ERROR")
spark = SparkSession.builder.appName("App").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

def stop_execution():
    sc.stop()
    exit(1)

# Check whether all necessary command line arguments are set
if len(sys.argv) < 4:
    print("Not all arguments are set correctly")
    print("Usage: $ spark.py <stages> <data_path> <model_dir>")
    print("Options for stages: 1 = train, 2 = eval & test")
    stop_execution()
STAGES = sys.argv[1]
DATA_PATH = sys.argv[2]
MODEL_DIR = sys.argv[3]
TRAIN, TEST = False, False

FEATURE_COLS = ["Month", "DayofMonth", "DepTimeT", "CRSDepTimeT", "CRSArrTimeT", "DepDelay", "CRSElapsedTime", "PunctualCarrier", "AverageCarrier"]

def preprocessing():
    print("Started data preprocessing")
    dataframe = spark.read.load(DATA_PATH, format="csv", sep=",", inferSchema="true", header="true")
    print(f"Loaded dataframe from {DATA_PATH}")

    dataframe = dataframe.drop(
        "ArrTime", 
        "ActualElapsedTime", 
        "AirTime", 
        "TaxiIn", 
        "Diverted", 
        "CarrierDelay", 
        "WeatherDelay", 
        "NASDelay", 
        "SecurityDelay", 
        "LateAircraftDelay",
        "FlightNum",
        "TailNum",
        "TaxiOut",
        "CancellationCode"
    )
    print("Dropped columns")

    dataframe = dataframe.filter(col("Cancelled") == 0).drop("Cancelled")
    print("Dropped rows of cancelled flights")
    dataframe = dataframe.filter(col("ArrDelay") != "NA")
    print("Dropped rows with missing arrival delays")

    dataframe = dataframe.withColumn("CRSElapsedTime", dataframe.CRSElapsedTime.cast(IntegerType())) \
                         .withColumn("ArrDelay", dataframe.ArrDelay.cast(IntegerType())) \
                         .withColumn("DepDelay", dataframe.DepDelay.cast(IntegerType()))
    print("Casted CRSElapsedTime, ArrDelay, and DepDelay to Integer")

    def convertToMinutes(hhmm):
        hhmm = str(hhmm).strip().zfill(4)
        if not hhmm.isdigit():
            print(f"Encountered invalid time: {hhmm}")
            return None
        mins = int(hhmm[-2:])
        hours = int(hhmm[:-2])
        return hours * 60 + mins

    convertToMinutesUDF = udf(convertToMinutes, IntegerType())

    dataframe = dataframe.withColumn("DepTimeT", convertToMinutesUDF(col("DepTime"))) \
                        .withColumn("CRSDepTimeT", convertToMinutesUDF(col("CRSDepTime"))) \
                        .withColumn("CRSArrTimeT", convertToMinutesUDF(col("CRSArrTime")))
    print("Transformed DepTime, CRSDepTime, and CRSArrTime")

    carriers = dataframe.select("UniqueCarrier").distinct().rdd.flatMap(lambda x: x).collect()
    punctual_carriers = ["HA", "AQ"]
    average_carriers = list(set(carriers) - set(punctual_carriers))
    dataframe = dataframe.withColumn("PunctualCarrier", when(col("UniqueCarrier").isin(punctual_carriers), 1).otherwise(0))
    dataframe = dataframe.withColumn("AverageCarrier", when(col("UniqueCarrier").isin(average_carriers), 1).otherwise(0))
    print("Applied one-hot encoding to UniqueCarrier based on two classes")

    dataframe = dataframe.drop("DepTime", "CRSDepTime", "CRSArrTime", "Distance", "UniqueCarrier", "DayOfWeek", "Origin", "Dest")
    print("Dropped obsolete columns")

    dataframe = dataframe.withColumn("label", col("ArrDelay")).drop("ArrDelay")
    print('Renamed column "ArrDelay" to "label"')

    print(f"Number of elements: {dataframe.count()}")
    print(f"Schema of final dataframe:\n{dataframe.schema.json()}")

    dataframe.write.parquet("preprocessed", mode="overwrite")
    print(f"Wrote preprocessed data to preprocessed")

    print("Finished data preprocessing")


def training():
    print("Starting model training")
    dataframe = spark.read.parquet("preprocessed")

    if not TEST:
        print("Only training in this step. Using all data for training")
        train_data = dataframe.select(*(FEATURE_COLS + ["label"]))
    else:
        print("Using 80-20 train-test split.")
        train_data, _ = dataframe.select(*(FEATURE_COLS + ["label"])).randomSplit([0.8, 0.2], seed=3)
    train_data.cache()

    vector_assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")
    lr = LinearRegression()
    pipeline_lr = Pipeline(stages=[vector_assembler, lr])
    model_lr = pipeline_lr.fit(train_data)
    model_lr.write().overwrite().save(os.path.join(MODEL_DIR, "lr"))
    print("Finished model training")


def testing():
    print("Starting model evaluation")
    dataframe = spark.read.parquet("preprocessed")
    available_model_paths = [f.path for f in os.scandir(MODEL_DIR) if f.is_dir()]

    if not available_model_paths:
        print(f"No models found in {MODEL_DIR}")
        stop_execution()

    if not TRAIN:
        print("Only testing in this step. Using all data for testing")
        test_data = dataframe.select(*(FEATURE_COLS + ["label"]))
    else:
        print("Using 80-20 train-test split.")
        _, test_data = dataframe.select(*(FEATURE_COLS + ["label"])).randomSplit([0.8, 0.2], seed=3)

    best_model, best_model_path = None, ""   
    lowest_rmse = float('inf')

    for available_model_path in available_model_paths:
        print(f"Evaluating model: {available_model_path}")
        loaded_model = PipelineModel.load(available_model_path)
        predictions = loaded_model.transform(test_data)

        evaluator = RegressionEvaluator()
        rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
        mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
        r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")

        if rmse < lowest_rmse:
            lowest_rmse = rmse
            best_model = loaded_model
            best_model_path = available_model_path

    print(f"Predicting with model: {best_model_path}")
    predictions = best_model.transform(test_data)
    predictions.drop("features").write.option("header", True).csv("predictions")


# Setup
if "1" in STAGES:
    TRAIN = True
if "2" in STAGES:
    TEST = True
if not os.path.isfile(DATA_PATH):
    print(f"{DATA_PATH}: File not found.")
    stop_execution()
print(f"Data path set to: {DATA_PATH}")
if not os.path.isdir(MODEL_DIR):
    print(f"{MODEL_DIR}: Directory not found.")
    stop_execution()
print(f"Model directory set to: {MODEL_DIR}")

preprocessing()

if TRAIN:
    training()
if TEST:
    testing()

sc.stop()
