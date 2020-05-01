package com.yash.sa2.pipeline.utility

import java.sql.{ DriverManager, PreparedStatement, Timestamp, Connection, ResultSet }
import com.yash.sa2.pipeline.constant.ConfigValues
import scala.util.{ Try, Failure, Success }
import scala.collection.mutable.ListBuffer
import java.util.ArrayList

/**
 * It acts as a utility object required for JDBC Operations
 *
 */
object JdbcUtility {

  /**
   * This close resources automatically
   * @param resource
   * @param block :function that uses the resource
   * @return :function that uses the resource
   */
  def autoClose[A <: AutoCloseable, B](resource: A)(block: A => B): B = {
    Try(block(resource)) match {
      case Success(result) =>
        resource.close()
        result
      case Failure(e) =>
        resource.close()
        throw e
    }
  }

  /**
   * This method returns the connection object
   * @param config :ConfigValues
   * @return Java Connection Object
   */
  def getDbCon(config: ConfigValues): Connection = {
    EtlUtility.notify("=====getDbCon=====")
    Class.forName(config.DbDriver)
    DriverManager.getConnection(config.DbUrl, config.DbUser, config.DbPass)
  }

  /**
   * This method simply executes the provided query in database
   * @param config: ConfigValues
   * @param query: String
   * @return queryRunStatus: Int
   */
  def executeQuery(config: ConfigValues, query: String): Int = {
    try {
      Class.forName(config.DbDriver)
      autoClose(getDbCon(config)) { con =>
        autoClose(con.prepareStatement(query)) { stmt =>
          val queryRunStatus = stmt.executeUpdate
          queryRunStatus
        }
      }
    } catch {
      case e: Exception =>
        e.printStackTrace
        val queryRunStatus = -1
        queryRunStatus
    }
  }

  def executeTestQuery(config: ConfigValues, query: String, tableName: String): (Boolean, String) = {
    EtlUtility.notify("Executing function executeTestQuery() - Start")
    var isSuccess = true
    var error = ""

    try {
      Class.forName("org.postgresql.Driver")
      autoClose(getDbCon(config)) { con =>
        autoClose(con.createStatement()) { stmt =>
          var rs = stmt.executeQuery(query)
          EtlUtility.notify("Records Available in Table: " + tableName + " =>")
          while (rs.next()) {
            EtlUtility.notify("Row: " + rs.getString(1))
            EtlUtility.notify("Row: " + rs.getString(2))
          }
        }
      }
    } catch {
      case e: Exception =>
        e.printStackTrace
        error = e.getMessage
        isSuccess = false
    }
    EtlUtility.notify("Executing function executeTestQuery() - Completed")
    (isSuccess, error)
  }
  def executeRulesQuery(config: ConfigValues, query: String): ArrayList[String] = {
    EtlUtility.notify("Executing function executeRulesQuery() - Start")
    var isSuccess = true
    var error = ""
    var result = new ArrayList[String]()
    try {
      Class.forName("org.postgresql.Driver")
      autoClose(getDbCon(config)) { con =>
        autoClose(con.createStatement()) { stmt =>
          var rs = stmt.executeQuery(query)
         while (rs.next()) {
            result.add(rs.getString(1))
          }
         }
      }
    } catch {
      case e: Exception =>
        e.printStackTrace
        error = e.getMessage
        isSuccess = false
    }
    EtlUtility.notify("Executing function executeRulesQuery() - Completed")
    result
  }
}