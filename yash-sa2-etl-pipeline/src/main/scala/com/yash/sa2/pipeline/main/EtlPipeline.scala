package com.yash.sa2.pipeline.main

import org.apache.spark.sql.{ SparkSession, DataFrame }
import java.util.Properties
import scala.util.Failure
import scala.util.Try
import java.sql.Timestamp
import scala.util.Success
import org.apache.spark.sql.functions.lit
import com.yash.sa2.pipeline.constant.ConfigValues
import org.postgresql.Driver
import com.yash.sa2.pipeline.utility.SparkUtility
import com.yash.sa2.pipeline.utility.EtlUtility
import com.yash.sa2.pipeline.utility.JdbcUtility
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.unix_timestamp
import org.apache.spark.sql.functions.{ to_date, to_timestamp }
import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import java.sql.{ DriverManager, PreparedStatement, Timestamp, Connection }
import scala.collection.mutable.ListBuffer
import org.apache.spark.sql.functions.udf
import java.time.format.DateTimeFormatter
import java.time.LocalDateTime
import java.util.ArrayList
/**
 * @purpose This object acts as main entry point
 */
object EtlPipeline {

  /**
   * Main ETL Job Entry Point
   * @param sparkSession
   * @param glueJobId
   * @param propFilePath
   */
  def run(sparkSession: SparkSession, glueJobId: String, propFilePath: String, module: String, sourceDeltaDf: DataFrame, dataSourceType: String): Unit = {

    val prop = EtlUtility.loadPropertiesFile(propFilePath)

    EtlUtility.getConfigObj(prop) match {
      case Some(config) => {

        EtlUtility.notify("Glue Job Started: " + glueJobId)
        EtlUtility.notify("Jar Version: " + "V1")

        var rawZonePath = config.rawZoneS3Path
        var transformedZonePath = config.transformedZoneS3Path
        var publishedZonePath = config.publishedZoneS3Path

        if (scala.util.Properties.envOrNone("s3") == Some("disable")) {
          rawZonePath = config.rawZoneLocalPath
          transformedZonePath = config.transformedZoneLocalPath
          publishedZonePath = config.publishedZoneLocalPath
        }

        EtlUtility.notify("rawZonePath: " + rawZonePath)
        EtlUtility.notify("transformedZonePath: " + transformedZonePath)
        EtlUtility.notify("publishedZonePath: " + publishedZonePath)
        EtlUtility.notify("DbUrl: " + config.DbUrl)
        EtlUtility.notify("DbUser: " + config.DbUser)

        if (module.equals("RawToTransformedS3_ETLJob")) {
          if (sourceDeltaDf != null) {
            processDataFromRawToTransformedS3Zone(sparkSession, glueJobId, propFilePath, sourceDeltaDf, transformedZonePath, publishedZonePath, config, dataSourceType)
          } else {
            EtlUtility.notify("Source Data Frame is empty.")
          }
        }

        if (module.equals("TransformedToPublishedS3_ETLJob")) {
          if (sourceDeltaDf != null) {
            processDataFromTransformedToPublishedS3Zone(sparkSession, glueJobId, propFilePath, sourceDeltaDf, transformedZonePath, publishedZonePath, config, dataSourceType)
          } else {
            EtlUtility.notify("Source Data Frame is empty.")
          }
        }
      }
      case None => EtlUtility.notify("Configuration arguments missing")
    }
  }

  /**
   * ETL Logic to process data from Raw to Transformed
   * @param sparkSession
   * @param glueJobId
   * @param propFilePath
   * @param rawZonePath
   * @param transformedZonePath
   * @param publishedZonePath
   * @param config
   */
  def processDataFromRawToTransformedS3Zone(sparkSession: SparkSession, glueJobId: String, propFilePath: String, rawDataDF: DataFrame, transformedZonePath: String, publishedZonePath: String, config: ConfigValues, dataSourceType: String): Unit = {

    EtlUtility.notify("Executing function processDataFromRawToTransformedS3Zone() - Start")
    var transformedDataDf: DataFrame = null
    transformedDataDf = SparkUtility.getTrimmedDataframe(rawDataDF)
    if (dataSourceType.equals("DS1")) {
      var result = JdbcUtility.executeRulesQuery(config, config.rawToTransformedQuery)
      var rs = result.iterator()
      var write_operations = new ArrayList[String]()
      while (rs.hasNext()) {
        var function_name = rs.next()
        //System.out.println(function_name + "=====================" + function_name.substring(0, 5).equals("write") + "===========================")
        if (function_name.substring(0, 5).equals("write")) {
          write_operations.add(function_name)
        } else {
          val method = SparkUtility.getClass.getMethod(function_name, transformedDataDf.getClass)
          var value = method.invoke(SparkUtility, transformedDataDf)
          transformedDataDf = value.asInstanceOf[DataFrame]
        }
      }
      if (!write_operations.isEmpty()) {
        var ws = write_operations.iterator()
        while (ws.hasNext()) {
          var write_function_name = ws.next()
          val method = EtlPipeline.getClass.getMethod(write_function_name, sparkSession.getClass(), transformedZonePath.getClass(), transformedDataDf.getClass(), config.getClass(), dataSourceType.getClass())
          method.invoke(EtlPipeline, sparkSession, transformedZonePath, transformedDataDf, config, dataSourceType)
        }
      } else {
        writeDataIntoTransformedZoneInCSV(sparkSession, transformedZonePath, transformedDataDf, config, dataSourceType)
      }
    }
    if (dataSourceType.equals("DS2")) {
      writeDataIntoTransformedZoneInParquet(sparkSession, transformedZonePath, transformedDataDf, config, dataSourceType)
    }
    EtlUtility.notify("Executing function processDataFromRawToTransformedS3Zone() - Completed")
  }

  def processDataFromTransformedToPublishedS3Zone(sparkSession: SparkSession, glueJobId: String, propFilePath: String, rawDataDF: DataFrame, transformedZonePath: String, publishedZonePath: String, config: ConfigValues, dataSourceType: String): Unit = {

    EtlUtility.notify("Executing function processDataFromTransformedToPublishedS3Zone() - Start")

    var transformedDataDf = rawDataDF

    var publishedDataDf: DataFrame = null

    if (dataSourceType.equals("DS1")) {
      transformedDataDf = SparkUtility.getTrimmedDataframe(transformedDataDf)

      var result = JdbcUtility.executeRulesQuery(config, config.transformedToPublishedQuery)
      var rs = result.iterator()
      var write_operations = new ArrayList[String]()
      var custom_operation = ""
      while (rs.hasNext()) {
        var function_name = rs.next()
        //System.out.println(function_name + "=====================" + function_name.substring(0, 5).equals("write") + "===========================")
        if (function_name.substring(0, 5).equals("write")) {
          write_operations.add(function_name)
        } else {
          val method = SparkUtility.getClass.getMethod(function_name, transformedDataDf.getClass)
          var value = method.invoke(SparkUtility, transformedDataDf)
          transformedDataDf = value.asInstanceOf[DataFrame]
        }
      }
      publishedDataDf = transformedDataDf
      publishedDataDf.printSchema()
      if (!write_operations.isEmpty()) {
        var ws = write_operations.iterator()
        while (ws.hasNext()) {
          var write_function_name = ws.next()
          val method = EtlPipeline.getClass.getMethod(write_function_name, sparkSession.getClass(), publishedZonePath.getClass(), publishedDataDf.getClass(), config.getClass(), dataSourceType.getClass())
          method.invoke(EtlPipeline, sparkSession, publishedZonePath, publishedDataDf, config, dataSourceType)
        }
      } else {
        writeDataIntoPublishedS3ZoneInCSV(sparkSession, publishedZonePath, publishedDataDf, config, dataSourceType)
      }
    }
    if (dataSourceType.equals("DS2")) {
      transformedDataDf = SparkUtility.getTrimmedDataframe(transformedDataDf)
      publishedDataDf = transformedDataDf
      writeDataIntoPublishedS3ZoneInCSV(sparkSession, publishedZonePath, publishedDataDf, config, dataSourceType)
    }
    EtlUtility.notify("Executing function processDataFromTransformedToPublishedS3Zone() - Completed")
  }

  /**
   * This function is used to Convert Required Column from String Type to Timestamp Type
   * @param outputDf
   * @param config
   * @param timestampColName
   * @return DataFrame
   */
  def convertStringToTimestampCol(outputDf: DataFrame, config: ConfigValues, timestampColName: String): DataFrame = {
    var updatedDF = outputDf
    updatedDF = updatedDF.withColumn(
      timestampColName,
      to_timestamp(unix_timestamp(updatedDF(timestampColName), config.supportedTimeStampFormatInJsonFile).cast(TimestampType)))
    updatedDF
  }

  /**
   * This function is used to Populate transformed Zone
   * @param sparkSession
   * @param transformedZonePath
   * @param outputDf
   * @param config
   */
  def writeDataIntoTransformedZoneInParquet(sparkSession: SparkSession, transformedZonePath: String, outputDf: DataFrame, config: ConfigValues, dataSourceType: String): Unit = {
    EtlUtility.notify("Writing data in Prarquet Form into Transformed Zone starts:")
    outputDf.printSchema()
    outputDf.show()
    outputDf.write.mode(SaveMode.Append).partitionBy("file_type").parquet(transformedZonePath + "/" + dataSourceType + "/")
    EtlUtility.notify("Writing data in Prarquet Form into Transformed Zone Completed!")
    //Reading Data From transformed Zone for Testing Purpose Only
    //readDataFromZone(sparkSession, transformedZonePath, "parquet", dataSourceType)
  }
  /**
   * This function is used to Populate transformed Zone
   * @param sparkSession
   * @param transformedZonePath
   * @param outputDf
   * @param config
   */
  def writeDataIntoTransformedZoneInCSV(sparkSession: SparkSession, transformedZonePath: String, outputDf: DataFrame, config: ConfigValues, dataSourceType: String): Unit = {
    EtlUtility.notify("Writing data in CSV Form into Transformed Zone starts:")
    outputDf.printSchema()
    outputDf.show()
    outputDf.coalesce(1).write.mode(SaveMode.Append).partitionBy("file_type").option("header", "true").csv(transformedZonePath + "/" + dataSourceType + "/")
    EtlUtility.notify("Writing data in CSV Form into Transformed Zone Completed!")
    //Reading Data From transformed Zone for Testing Purpose Only
    //readDataFromZone(sparkSession, transformedZonePath, "csv", dataSourceType)
  }
  /**
   * This function is used to Populate published Zone
   * @param sparkSession
   * @param publishedZonePath
   * @param outputDf
   * @param config
   */
  def writeDataIntoPublishedS3ZoneInParquet(sparkSession: SparkSession, publishedZonePath: String, outputDf: DataFrame, config: ConfigValues, dataSourceType: String): Unit = {
    EtlUtility.notify("Writing data into Published Zone starts:")
    outputDf.printSchema()
    outputDf.show()
    outputDf.write.mode(SaveMode.Append).partitionBy("file_type").parquet(publishedZonePath + "/" + dataSourceType + "/")
    EtlUtility.notify("Writing data Completed!")
    // Reading Data From published S3 Zone for Testing Purpose Only
    //readDataFromZone(sparkSession, publishedZonePath, "csv", dataSourceType)
  }
  /**
   * This function is used to Populate published Zone
   * @param sparkSession
   * @param publishedZonePath
   * @param outputDf
   * @param config
   */
  def writeDataIntoPublishedS3ZoneInCSV(sparkSession: SparkSession, publishedZonePath: String, outputDf: DataFrame, config: ConfigValues, dataSourceType: String): Unit = {
    EtlUtility.notify("Writing data into Published Zone starts:")
    outputDf.printSchema()
    outputDf.show()
    outputDf.coalesce(1).write.mode(SaveMode.Append).partitionBy("file_type").option("header", "true").csv(publishedZonePath + "/" + dataSourceType + "/")
    EtlUtility.notify("Writing data Completed!")
    // Reading Data From published S3 Zone for Testing Purpose Only
    //readDataFromZone(sparkSession, publishedZonePath, "parquet", dataSourceType)
  }
  /**
   * This function is used to Read Data From S3 Zone
   * @param sparkSession
   * @param zonePath
   */
  def readDataFromZone(sparkSession: SparkSession, zonePath: String, format: String, dataSourceType: String): Unit = {
    EtlUtility.notify("Reading Zone: " + zonePath)
    var df: DataFrame = null
    if (format == "csv")
      df = sparkSession.read.option("header", "true").csv(zonePath + "/" + dataSourceType + "/")
    else if (format == "parquet")
      df = sparkSession.read.parquet(zonePath + "/" + dataSourceType + "/")
    EtlUtility.notify("Schema: ")
    df.printSchema()
    EtlUtility.notify("Records count: " + df.count())
    df.show()
  }
  /**
   * This function is used to Write Data From S3 Zone to Database Tables
   * @param sparkSession
   * @param outputDf
   * @param config
   */
  def writeTransformedDataIntoRDS(sparkSession: SparkSession, zonePath: String, outputDf: DataFrame, config: ConfigValues, dataSourceType: String): Unit = {
    var table = "yash_sa_schema." + config.transformedZoneTableName
    val url = config.DbUrl
    val prop = new java.util.Properties
    prop.setProperty("driver", config.DbDriver)
    prop.setProperty("user", config.DbUser)
    prop.setProperty("password", config.DbPass)
    outputDf.write.mode(SaveMode.Append).jdbc(url, table, prop)
  }
  /**
   * This function is used to Write Data From S3 Zone to Database Tables
   * @param sparkSession
   * @param outputDf
   * @param config
   */
  def writePublishedDataIntoRDS(sparkSession: SparkSession, zonePath: String, outputDf: DataFrame, config: ConfigValues, dataSourceType: String): Unit = {
    var table = "yash_sa_schema." + config.publishedZoneTableName
    val url = config.DbUrl
    val prop = new java.util.Properties
    prop.setProperty("driver", config.DbDriver)
    prop.setProperty("user", config.DbUser)
    prop.setProperty("password", config.DbPass)
    outputDf.write.mode(SaveMode.Append).jdbc(url, table, prop)
  }
}