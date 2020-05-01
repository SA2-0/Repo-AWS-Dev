package com.yash.sa2.pipeline.utility

import org.apache.spark.sql.{ SparkSession, DataFrame, SaveMode }
import org.apache.spark.sql.types.{ StructType, StructField }
import org.apache.spark.sql.functions.{ lit, trim }
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.StructType
import scala.collection.immutable._
import scala.collection.mutable.ListBuffer
import com.yash.sa2.pipeline.constant.ConfigValues
import java.lang.Double
import java.util.Date
import java.text.SimpleDateFormat
import java.sql.Timestamp
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.expressions.Window

/**
 * It acts as a utility object required for Spark Operations
 *
 */
object SparkUtility {

  val dateFormat = new SimpleDateFormat("yyyy-MM-dd")

  def getTrimmedDataframe(rawOutputDataDf: DataFrame): DataFrame = {
    var OutputDataDf = rawOutputDataDf
    val fileSchema = OutputDataDf.schema
    fileSchema.foreach { fileStructField =>
      if (fileStructField.dataType.typeName.toLowerCase().equals("string")) {
        OutputDataDf = OutputDataDf.withColumn(fileStructField.name, trim(OutputDataDf(fileStructField.name)))
      }
    }
    //System.out.println("=========================Completed the function getTrimmedDataframe=============================")
    OutputDataDf
  }

  def cacheTempTable(tempTableName: String, sparkSession: SparkSession): Unit = {
    sparkSession.catalog.cacheTable(tempTableName)
  }

  def uncacheTempTable(tempTableName: String, sparkSession: SparkSession): Unit = {
    sparkSession.catalog.uncacheTable(tempTableName)
  }

  def convertStringToDate(value: String): Date = {
    val dataCaptureDate = dateFormat.parse(value)
    dataCaptureDate
  }
  def dropNull(rawOutputDataDf: DataFrame): DataFrame = {
    var transformedDataDf = rawOutputDataDf
    //var transformedDataDf = rawOutputDataDf.na.drop()
    transformedDataDf
  }

  def customModel(rawOutputDataDf: DataFrame): DataFrame = {
    var transformedDataDf = rawOutputDataDf
    transformedDataDf = transformedDataDf.drop("col64").drop("_c64").withColumn("orderreceiveddatev1", concat((substring(col("orderreceiveddate"), 7, 4)), lit("-"), (substring(col("orderreceiveddate"), 4, 2)), lit("-"), (substring(col("orderreceiveddate"), 1, 2)))).withColumn("custacknwldgmntdatev1", concat((substring(col("custacknwldgmntdate"), 7, 4)), lit("-"), (substring(col("custacknwldgmntdate"), 4, 2)), lit("-"), (substring(col("custacknwldgmntdate"), 1, 2))))

    transformedDataDf = transformedDataDf.withColumn("orderreceiveddatev1", col("orderreceiveddatev1").cast(DateType)).withColumn("custacknwldgmntdatev1", col("custacknwldgmntdatev1").cast(DateType))

    transformedDataDf = transformedDataDf.withColumn("SLA", datediff(col("custacknwldgmntdatev1"), col("orderreceiveddatev1"))).withColumn("UniqueId", concat((col("marketinggroup").cast(IntegerType)), col("billtocode"), col("ordernumber"))).withColumn("Discount", (col("ordergrossvalue") - col("ordernetvalue"))).withColumn("Adj_OrderGrossValue", col("orderGrossValue") / 71).withColumn("Adj_OrderNetValue", col("ordernetvalue") / 71)

    transformedDataDf = transformedDataDf.drop("col64", "_c64", "BuildPlantNumber", "ReclearPlantNumber", "BilltoPO", "Nomenclature", "SIONumber",
      "DiscountAuthNumber", "ContainerControlNbr", "BillToAddressLine1", "BillToAddressLine2", "BillToAddressLine3",
      "BillToAddressLine4", "BillToAddressLine5", "BillToPostalCode", "BillToCountryCode", "BillToCountyCode",
      "BillToCounty", "BillToExtActiveInactiveStatus", "ShipToCode", "EndCustOrShipTo", "ShipToAddressLine1",
      "ShipToAddressLine2", "ShipToAddressLine3", "ShipToAddressLine4", "ShipToAddressLine5", "ShipToZipCode",
      "ShipToCountryCode", "ShipToCountyCode", "custacknwldgmntdatev1", "orderreceiveddatev1",
      "OrderRequestNumber", "SystemOrigin", "WorkUnitNumber", "WorkUnitRevNbr", "ActualCTSDate",
      "CustAcknwldgmntDate", "ScheduledBuildDate", "CurrEstCTSDate")

    val countryUdf = udf { country: String => country.replaceAll("[.]", "").toLowerCase().capitalize }

    transformedDataDf = transformedDataDf.withColumn("Country", countryUdf(col("BillToCountryName")))

    val dateUdf = udf { dt: String => dt.replaceAll("[/]", "-") }

    transformedDataDf = transformedDataDf.withColumn("OrderReceivedDate", dateUdf(col("OrderReceivedDate"))).withColumn("ActualSHPDate", dateUdf(col("ActualSHPDate")))

    transformedDataDf = transformedDataDf.withColumn("OrderReceivedDate", date_format(to_date(col("OrderReceivedDate"), "dd-MM-yyyy"), "MM/dd/yyyy")).withColumn("ActualSHPDate", date_format(to_date(col("ActualSHPDate"), "dd-MM-yyyy"), "MM/dd/yyyy"))

    val Europe = List("Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus", "Belgium", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Estonia", "Finland", "France", "Georgia", "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Italy", "Latvia", "Liechtenstein", "Lithuania", "Luxembourg", "Macedonia", "Malta", "Moldova", "Monaco", "Montenegro", "Netherlands", "Norway", "Poland", "Portugal", "Romania", "San Marino", "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland", "Ukraine", "United Kingdom", "Vatican City")

    val Asia = List("Afghanistan", "Bahrain", "Bangladesh", "Bhutan", "Brunei", "Myanmar", "Cambodia", "China", "East Timor", "India", "Indonesia", "Iran", "Iraq", "Israel", "Japan", "Jordan", "Kazakhstan", "Korea, North", "Korea, South", "Kuwait", "Kyrgyzstan", "Laos", "Lebanon", "Malaysia", "Maldives", "Mongolia", "Nepal", "Oman", "Pakistan", "Philippines", "Qatar", "Russian Federation", "Saudi Arabia", "Singapore", "Sri Lanka", "Syria", "Tajikistan", "Thailand", "Turkey", "Turkmenistan", "United Arab Emirates", "Uzbekistan", "Vietnam", "Yemen")

    val Africa = List("Algeria", "Angola", "Benin", "Botswana", "Burkina", "Burundi", "Cameroon", "Cape Verde", "Central African Republic", "Chad", "Comoros", "Congo", "Congo, Democratic Republic of", "Djibouti", "Egypt", "Equatorial Guinea", "Eritrea", "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Ivory Coast", "Kenya", "Lesotho", "Liberia", "Libya", "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria", "Rwada", "Sao Tome and Principe", "Senegal", "Seychelles", "Sierra Leone", "Somalia", "South Africa", "South Sudan", "Sudan", "Swaziland", "Tanzania", "Togo", "Tunisia", "Uganda", "Zambia", "Zimbabwe")

    val NorthAmerica = List("Antigua and Barbuda", "Bahamas", "Barbados", "Belize", "Canada", "Costa Rica", "Cuba", "Dominica", "Dominican Republic", "El Salvador", "Grenada", "Guatemala", "Haiti", "Honduras", "Jamaica", "Mexico", "Nicaragua", "Panama", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Trinidad and Tobago", "United States", "U.S.A.")

    val Oceania = List("Australia", "Fiji", "Kiribati", "Marshall Islands", "Micronesia", "Nauru", "New Zealand", "Palau", "Papua New Guinea", "Samoa", "Solomon Islands", "Tonga", "Tuvalu", "Vanuatu")

    val SouthAmerica = List("Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador", "Guyana", "Paraguay", "Peru", "Suriname", "Uruguay", "Venezuela")

    transformedDataDf = transformedDataDf.withColumn("Region", when(col("billtocountryname") isin (Asia: _*), "Asia").when(col("billtocountryname") isin (Europe: _*), "Europe").when(col("billtocountryname") isin (Africa: _*), "Africa").when(col("billtocountryname") isin (NorthAmerica: _*), "NorthAmerica").when(col("billtocountryname") isin (Oceania: _*), "Oceania").when(col("billtocountryname") isin (SouthAmerica: _*), "SouthAmerica"))

    transformedDataDf = transformedDataDf.select(col("UniqueId").as("UniqueId"), col("marketinggroup").as("MarketingGroup"), col("ordernumber").as("OrderNumber"), col("ordertype").as("OrderType"),
      col("OrderReceivedDate").as("OrderReceivedDate"), col("seriescode").as("SeriesCode"), col("modelcode").as("ModelCode"), col("ordergrossvalue").as("OrderGrossValue"),
      col("ordernetvalue").as("OrderNetValue"), col("Discount").as("Discount"), col("Adj_OrderGrossValue").as("Adj_OrderGrossValue"), col("Adj_OrderNetValue").as("Adj_OrderNetValue"),
      col("currencycode").as("CurrencyCode"), col("unitserialnumber").as("UnitSerialNumber"), col("ActualSHPDate").as("ActualSHPDate"), col("endcustomerorder").as("EndCustomerOrder"),
      col("itatruckclass").as("ITATruckClass"), col("billtocode").as("BillToCode"), col("billtoname").as("BillToName"), col("billtobrand").as("BillToBrand"),
      col("billtocustomertype").as("BillToCustomerType"), col("billtocustomertypedesc").as("BillToCustomerTypeDesc"), col("billtodualbrand").as("BillToDualBrand"),
      col("shiptoname").as("ShipToName"), col("shiptocountryname").as("ShipToCountryName"), col("billtocountryname").as("BillToCountryName"), col("Country").as("Country"),
      col("Region").as("Region"), col("billtocity").as("BillToCity"), col("billtostate").as("BillToState"), col("shiptostandardindustrycode").as("ShipToStandardIndustryCode"),
      col("shiptocity").as("ShipToState"), col("shiptostate").as("ShipToCity"), col("orderingdealer").as("OrderingDealer"), col("deliverymethod").as("DeliveryMethod"),
      col("SLA").as("SLA"), col("file_type").as("file_type"))

    val w = Window.orderBy("UniqueId")

    transformedDataDf = transformedDataDf.withColumn("S.No", row_number().over(w))
    transformedDataDf
  }

}