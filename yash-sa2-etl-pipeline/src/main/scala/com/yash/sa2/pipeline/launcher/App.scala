package com.yash.sa2.pipeline.launcher

import org.apache.spark.sql.SparkSession
import org.apache.log4j.{ Level, Logger }
import java.util.UUID
import com.yash.sa2.pipeline.main.EtlPipeline
import org.apache.spark.sql.{ SparkSession, DataFrame }

/**
 * @purpose Local Launcher App
 *
 */
object App {

  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "D:/WorkSpaces/Hadoop/hadoop-common-2.2.0-bin-master");
    val sparkSession = SparkSession.builder.master("local[*]").appName(this.getClass.getSimpleName)
      .config("spark.ui.showConsoleProgress", "false").getOrCreate();

    // set error level
    sparkSession.sparkContext.setLogLevel("ERROR")
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)

    val glueJobId = UUID.randomUUID().toString
    val propFilePath = args(0)
    val module = "RawToTransformedS3_ETLJob"
//    val module = "TransformedToPublishedS3_ETLJob"
   
    
    
    var raw_zone_path_map = Map[String, String]()
    raw_zone_path_map += ("DS1" -> "D:/PTG/Accelerators/SA2.0/S3/ptg-banking-poc/sa_raw_zone/DS1")
    raw_zone_path_map += ("DS2" -> "D:/PTG/Accelerators/SA2.0/S3/ptg-banking-poc/sa_raw_zone/DS2")
    
    var transformed_zone_path_map = Map[String, String]()
    transformed_zone_path_map += ("DS1" -> "D:/PTG/Accelerators/SA2.0/S3/ptg-banking-poc/sa_transformed_zone/DS1")
    transformed_zone_path_map += ("DS2" -> "D:/PTG/Accelerators/SA2.0/S3/ptg-banking-poc/sa_transformed_zone/DS2")
    
    var spark_source_delta_df: DataFrame = null
    if (module.equals("RawToTransformedS3_ETLJob")) {
      raw_zone_path_map.foreach { case (dataSourceType, folderPath) =>
        println("Data Source Type: " + dataSourceType)
        spark_source_delta_df = sparkSession.read.option("header", "true").option("inferSchema", "true").csv(folderPath)
        spark_source_delta_df.printSchema()
        spark_source_delta_df.show()
        EtlPipeline.run(sparkSession, glueJobId, propFilePath, module, spark_source_delta_df, dataSourceType)
      }
    } else if (module.equals("TransformedToPublishedS3_ETLJob")) {
      transformed_zone_path_map.foreach { case (dataSourceType, folderPath) =>
        println("Data Source Type: " + dataSourceType) 
        spark_source_delta_df = sparkSession.read.option("mergeSchema", "true").parquet(folderPath)
        spark_source_delta_df.printSchema()
        spark_source_delta_df.show()
        EtlPipeline.run(sparkSession, glueJobId, propFilePath, module, spark_source_delta_df, dataSourceType)
      }
    }

    sparkSession.stop
  }

}


