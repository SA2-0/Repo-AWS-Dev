package com.yash.sa2.pipeline.constant

import java.util.Properties
import scala.collection.mutable.ListBuffer


/**
 * @purpose This object contains the properties in String keys & values format that should be present in the configuration file also
 *
 */
case class ConfigValues(prop: Properties) {

  val DbUrl = prop.getProperty("DbUrl").trim

  val DbUser = prop.getProperty("DbUser").trim
  val DbPass = prop.getProperty("DbPass").trim

  val DbDriver = prop.getProperty("DbDriver").trim
  val SecretKey = prop.getProperty("SecretKey").trim
  val EncAlgorithm = prop.getProperty("EncAlgorithm").trim

  val jobPrcssStat_InProgress = prop.getProperty("jobPrcssStat_InProgress").trim
  val jobPrcssStat_Completed = prop.getProperty("jobPrcssStat_Completed").trim
  val jobPrcssStat_Failed = prop.getProperty("jobPrcssStat_Failed").trim

  val rawZoneLocalPath = prop.getProperty("rawZoneLocalPath").trim
  val rawZoneS3Path = prop.getProperty("rawZoneS3Path").trim

  val transformedZoneLocalPath = prop.getProperty("transformedZoneLocalPath").trim
  val transformedZoneS3Path = prop.getProperty("transformedZoneS3Path").trim

  val publishedZoneLocalPath = prop.getProperty("publishedZoneLocalPath").trim
  val publishedZoneS3Path = prop.getProperty("publishedZoneS3Path").trim
  
  val supportedTimeStampFormatInJsonFile = prop.getProperty("supportedTimeStampFormatInJsonFile").trim
  
  val rawToTransformedQuery = prop.getProperty("rawToTransformedQuery").trim
  val transformedToPublishedQuery = prop.getProperty("transformedToPublishedQuery").trim
  
  val transformedZoneTableName= prop.getProperty("transformedZoneTableName")
  val publishedZoneTableName=prop.getProperty("publishedZoneTableName")
}