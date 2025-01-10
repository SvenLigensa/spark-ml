import os
import logging
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("app.log")]
)
logger = logging.getLogger("Spark Application")

# Initialize Spark
conf = SparkConf()
conf.setAppName("Spark Application")
conf.setMaster("local")
conf.set("spark.hadoop.fs.defaultFS", "file:///")
sc = SparkContext.getOrCreate(conf)
sc.setLogLevel("ERROR")
spark = SparkSession.builder.appName("App").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Check whether all necessary command line arguments are set
if len(sys.argv) < 3:
    logger.error("Not all arguments are set correctly")
    logger.error("Usage: $ spark.py <data_path> <model_dir>")
    exit(1)
DATA_PATH = sys.argv[1]
MODEL_DIR = sys.argv[2]
DATA_DIR, TRAIN_DATA_DIR, TEST_DATA_DIR = None

def setup():
    logger.info(f"Data path set to: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        logger.error(f"{DATA_PATH}: Not found.")
        exit(1)
    if os.path.isdir(DATA_PATH):
        DATA_DIR = DATA_PATH
        train_dir = os.path.join(DATA_PATH, "train")
        test_dir = os.path.join(DATA_PATH, "test")
        if os.path.exists(train_dir) and os.path.exists(test_dir):
            TRAIN_DATA_DIR = train_dir
            TEST_DATA_DIR = test_dir
            logger.info(f"{DATA_PATH}: Subdirectories train/ and test/ found. Using given split.")
        else:
            logger.info(f"{DATA_PATH}: No train-test-split found. Using random split.")
    elif os.path.isfile(DATA_PATH):
        DATA_DIR = os.path.dirname(DATA_PATH)
        logger.info(f"{DATA_PATH}: Got a single file. Using random split.")

    if not os.path.exists(MODEL_DIR) or not os.path.isdir(MODEL_DIR):
        logger.info(f"{MODEL_DIR}: Not found or no directory.")
    else:
        logger.info(f"Model directory set to: {MODEL_DIR}")


def preprocessing():
    logger.info("Started data preprocessing")
    dataframe = spark.read.load(DATA_PATH, format="csv", sep=",", inferSchema="true", header="true")
    logger.info(f"Loaded dataframe from {DATA_PATH}")

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
    logger.debug("Dropped columns")

    dataframe = dataframe.filter(col("Cancelled") == 0).drop("Cancelled")
    logger.debug("Dropped rows of cancelled flights")
    dataframe = dataframe.filter(col("ArrDelay") != "NA")
    logger.debug("Dropped rows with missing arrival delays")

    dataframe = dataframe.withColumn("CRSElapsedTime", dataframe.CRSElapsedTime.cast(IntegerType())) \
                        .withColumn("ArrDelay", dataframe.ArrDelay.cast(IntegerType())) \
                        .withColumn("DepDelay", dataframe.DepDelay.cast(IntegerType()))
    logger.debug("Casted CRSElapsedTime, ArrDelay, and DepDelay to Integer")

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
    logger.debug("Transformed DepTime, CRSDepTime, and CRSArrTime")

    carriers = dataframe.select("UniqueCarrier").distinct().rdd.flatMap(lambda x: x).collect()
    punctual_carriers = ["HA", "AQ"]
    average_carriers = list(set(carriers) - set(punctual_carriers))
    dataframe = dataframe.withColumn("PunctualCarrier", when(col("UniqueCarrier").isin(punctual_carriers), 1).otherwise(0))
    dataframe = dataframe.withColumn("AverageCarrier", when(col("UniqueCarrier").isin(average_carriers), 1).otherwise(0))
    logger.debug("Applied one-hot encoding to UniqueCarrier based on two classes")

    dataframe = dataframe.drop("DepTime", "CRSDepTime", "CRSArrTime", "Distance", "UniqueCarrier", "DayOfWeek", "Origin", "Dest")
    logger.debug("Dropped obsolete columns")

    logger.info(f"Number of elements: {dataframe.count()}")
    logger.info(f"Schema of final dataframe:\n{dataframe.schema.json()}")

    print(DATA_DIR)
    output_path = os.path.join(DATA_DIR, "preprocessed")
    dataframe.write.parquet(output_path, mode="overwrite")
    logger.info(f"Wrote dataframe to {output_path}")

    logger.info("Finished data preprocessing")


def training():
    pass


def evaluation():
    logger.info("Starting model evaluation")
    print("Starting model evaluation")

    available_model_paths = [f.path for f in os.scandir(MODEL_DIR) if f.is_dir()]

    for available_model_path in available_model_paths:
        print(available_model_path)
        loaded_model = PipelineModel.load(available_model_path)

        dataframe = spark.read.parquet(INPUT_DATA_PATH)
        dataframe = dataframe.withColumn("label", col("ArrDelay")).drop("ArrDelay")

        FEATURE_COLS = ["Month", "DayofMonth", "DepTimeT", "CRSDepTimeT", "CRSArrTimeT", "DepDelay", "CRSElapsedTime", "PunctualCarrier", "AverageCarrier"]
        _, test_data = dataframe.select(*(FEATURE_COLS + ["label"])).randomSplit([0.8, 0.2], seed=3)

        predictions = loaded_model.transform(test_data)

        print("Evaluating...")

        evaluator = RegressionEvaluator()
        rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
        mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
        r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")





#print("Subdirectories:")
#for subdir in subdirectories:
#    print(subdir)


def testing():
    pass



evaluation()



sc.stop()
