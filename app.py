from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("app.log")]
)
logger = logging.getLogger("Spark Application")

conf = SparkConf()
conf.setAppName("Spark Application")
conf.setMaster("local")
conf.set("spark.hadoop.fs.defaultFS", "file:///")
sc = SparkContext.getOrCreate(conf)
sc.setLogLevel("ERROR")
spark = SparkSession.builder.appName("App").getOrCreate()
spark.sparkContext.setLogLevel("WARN")


# The input and output path are command line arguments
if len(sys.argv) < 3:
    logger.error("Input and output path must be set. Usage: $ spark.py <input_path> <output_path>")
    exit(1)
INPUT_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2]

logger.info(f"Data will be read from {INPUT_PATH}")
logger.info(f"Data will be written to {OUTPUT_PATH}")

dataframe = spark.read.load(INPUT_PATH, format="csv", sep=",", inferSchema="true", header="true")
logger.info(f"Loaded dataframe from {INPUT_PATH}")

# Drop the features for which the values are not available at the time of departure
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
    hhmm = str(hhmm).strip().zfill(4)  # Ensure it is a string and remove leading/trailing spaces
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

dataframe.write.parquet(OUTPUT_PATH, mode="overwrite")
logger.info(f"Wrote dataframe to {OUTPUT_PATH}")

sc.stop()
