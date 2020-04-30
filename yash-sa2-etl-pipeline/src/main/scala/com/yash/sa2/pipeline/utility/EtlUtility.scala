package com.yash.sa2.pipeline.utility

import java.util.Properties
import java.io.FileInputStream
import com.yash.sa2.pipeline.constant.ConfigValues
import java.sql.Timestamp

/**
 * It acts as a utility object required for ETL Operations
 *
 */
object EtlUtility {

  /**
   * This method return 'java.util.Properties' object
   * for provided property file path
   * @param propFilePath
   * @return Java Properties Object
   */
  def loadPropertiesFile(propFilePath: String): Properties = {
    val config = new Properties()
    config.load(new FileInputStream(propFilePath))
    config
  }

  /**
   * This method prepare connectionProperties object required for connecting to database
   * @param config :ConfigValues Object
   * @return Java Properties Object
   */
  def prepareConProperties(config: ConfigValues): Properties = {
    val connectionProperties = new Properties()
    connectionProperties.put("user", config.DbUser)
    connectionProperties.put("password", config.DbPass)
    connectionProperties.put("driver", config.DbDriver)
    connectionProperties
  }

  /**
   * This method return the ConfigValues object for provided property object
   * @param prop
   * @return Option(ConfigValues)
   */
  def getConfigObj(prop: Properties): Option[ConfigValues] = {
    try {
      Some(ConfigValues(prop))
    } catch {
      case ex: NullPointerException => None
    }
  }

  /**
   * This method notify the user for provided message
   * currently prints on console
   * @param message
   */
  def notify(message: String): Unit = {
    val currentTs = new Timestamp(System.currentTimeMillis)
    val msg = "## " + currentTs + " : " + message
    println(msg)
  }

  /**
   * The function is use to dynamically get any property
   * @param config: ConfigValues
   * @param configField: String
   * @param configFieldValue: String
   */
  def getConfigValueDynamically(config: ConfigValues, configField: String): String = {
    var configFieldValue: String = ""
    config.getClass.getDeclaredFields foreach { f =>
      f.setAccessible(true)
      if (f.getName == configField) {
        configFieldValue = f.get(config).toString
      }
    }
    configFieldValue
  }

}