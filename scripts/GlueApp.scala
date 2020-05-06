import com.amazonaws.services.glue.GlueContext
import com.amazonaws.services.glue.util.{GlueArgParser, Job, JsonOptions}
import org.apache.spark.SparkContext
import scala.collection.JavaConverters._
import com.amazonaws.services.glue.ChoiceOption
import com.amazonaws.services.glue.MappingSpec
import com.amazonaws.services.glue.ResolveSpec
import com.amazonaws.services.glue.errors.CallSite
import com.amazonaws.services.glue.util.GlueArgParser
import com.amazonaws.services.glue.util.Job
import com.amazonaws.services.glue.util.JsonOptions
import org.apache.spark.sql.{ SparkSession, DataFrame }
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.unix_timestamp
import org.apache.spark.sql.functions.{ to_date, to_timestamp }
import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import java.lang.Exception

import com.yash.sa2.pipeline.main.EtlPipeline

/**
 * @purpose Glue Launcher App
 *
 */
object GlueApp {
  
  /**
   * Main Function
   * @param sysArgs
   */
  def main(sysArgs: Array[String]) {
    
    // Getting Spark Session from Glue Context
    val spark: SparkContext = new SparkContext()
    val glueContext: GlueContext = new GlueContext(spark)
    val sparkSession = glueContext.getSparkSession

    val args = GlueArgParser.getResolvedOptions(sysArgs, Seq("JOB_NAME", "MODULE").toArray)
    Job.init(args("JOB_NAME"), glueContext, args.asJava)

    // Getting Glue Job Run ID
    val glueJobId = Job.runId.toString

    // Reference property file in Glue Job
    val propFilePath = "config.properties"
    
    val module = args("MODULE")

    var spark_source_delta_df: DataFrame = null
    if (module.equals("RawToTransformedS3_ETLJob")) {
      try {
        
        val datasource0 = glueContext.getCatalogSource(database = "sa_raw_db", tableName = "ds1", redshiftTmpDir = "", additionalOptions = JsonOptions("""{"mergeSchema": "true"}"""), transformationContext = "datasource0").getDynamicFrame()
        val applymapping0 = datasource0.applyMapping(mappings = Seq(("marketinggroup", "long", "marketinggroup", "long"), ("orderingdealer", "long", "orderingdealer", "long"), ("orderrequestnumber", "long", "orderrequestnumber", "long"), ("systemorigin", "string", "systemorigin", "string"), ("ordernumber", "long", "ordernumber", "long"), ("ordertype", "string", "ordertype", "string"), ("orderreceiveddate", "string", "orderreceiveddate", "string"), ("deliverymethod", "string", "deliverymethod", "string"), ("seriescode", "string", "seriescode", "string"), ("modelcode", "string", "modelcode", "string"), ("ordergrossvalue", "double", "ordergrossvalue", "double"), ("ordernetvalue", "double", "ordernetvalue", "double"), ("currencycode", "string", "currencycode", "string"), ("workunitnumber", "long", "workunitnumber", "long"), ("workunitrevnbr", "long", "workunitrevnbr", "long"), ("unitserialnumber", "string", "unitserialnumber", "string"), ("scheduledbuilddate", "string", "scheduledbuilddate", "string"), ("currestctsdate", "string", "currestctsdate", "string"), ("actualshpdate", "string", "actualshpdate", "string"), ("actualctsdate", "string", "actualctsdate", "string"), ("custacknwldgmntdate", "string", "custacknwldgmntdate", "string"), ("buildplantnumber", "long", "buildplantnumber", "long"), ("reclearplantnumber", "long", "reclearplantnumber", "long"), ("endcustomerorder", "string", "endcustomerorder", "string"), ("itatruckclass", "long", "itatruckclass", "long"), ("billtopo", "string", "billtopo", "string"), ("nomenclature", "string", "nomenclature", "string"), ("sionumber", "long", "sionumber", "long"), ("discountauthnumber", "long", "discountauthnumber", "long"), ("containercontrolnbr", "string", "containercontrolnbr", "string"), ("billtocode", "long", "billtocode", "long"), ("billtoname", "string", "billtoname", "string"), ("billtobrand", "string", "billtobrand", "string"), ("billtoaddressline1", "string", "billtoaddressline1", "string"), ("billtoaddressline2", "string", "billtoaddressline2", "string"), ("billtoaddressline3", "string", "billtoaddressline3", "string"), ("billtoaddressline4", "string", "billtoaddressline4", "string"), ("billtoaddressline5", "string", "billtoaddressline5", "string"), ("billtocity", "string", "billtocity", "string"), ("billtostate", "string", "billtostate", "string"), ("billtopostalcode", "string", "billtopostalcode", "string"), ("billtocountrycode", "string", "billtocountrycode", "string"), ("billtocountryname", "string", "billtocountryname", "string"), ("billtocountycode", "long", "billtocountycode", "long"), ("billtocounty", "string", "billtocounty", "string"), ("billtoextactiveinactivestatus", "string", "billtoextactiveinactivestatus", "string"), ("billtocustomertype", "long", "billtocustomertype", "long"), ("billtocustomertypedesc", "string", "billtocustomertypedesc", "string"), ("billtodualbrand", "string", "billtodualbrand", "string"), ("shiptocode", "long", "shiptocode", "long"), ("endcustorshipto", "string", "endcustorshipto", "string"), ("shiptoname", "string", "shiptoname", "string"), ("shiptoaddressline1", "string", "shiptoaddressline1", "string"), ("shiptoaddressline2", "string", "shiptoaddressline2", "string"), ("shiptoaddressline3", "string", "shiptoaddressline3", "string"), ("shiptoaddressline4", "string", "shiptoaddressline4", "string"), ("shiptoaddressline5", "string", "shiptoaddressline5", "string"), ("shiptocity", "string", "shiptocity", "string"), ("shiptostate", "string", "shiptostate", "string"), ("shiptozipcode", "string", "shiptozipcode", "string"), ("shiptocountrycode", "string", "shiptocountrycode", "string"), ("shiptocountryname", "string", "shiptocountryname", "string"), ("shiptocountycode", "string", "shiptocountycode", "string"), ("shiptostandardindustrycode", "long", "shiptostandardindustrycode", "long"), ("file_type", "string", "file_type", "string")), caseSensitive = false, transformationContext = "applymapping0")
        val recordCount0 = applymapping0.count
        println("ds1 table record count: " + recordCount0)
        if(recordCount0 > 0) {
            spark_source_delta_df = applymapping0.toDF()
            println("ds1 table Schema: ")
            spark_source_delta_df.printSchema()
            EtlPipeline.run(sparkSession, glueJobId, propFilePath, module, spark_source_delta_df, "DS1")
        }
        
        val datasource1 = glueContext.getCatalogSource(database = "sa_raw_db", tableName = "ds2", redshiftTmpDir = "", transformationContext = "datasource1").getDynamicFrame()
        val applymapping1 = datasource1.applyMapping(mappings = Seq(("rownumber", "long", "rownumber", "long"), ("customerid", "long", "customerid", "long"), ("surname", "string", "surname", "string"), ("creditscore", "long", "creditscore", "long"), ("geography", "string", "geography", "string"), ("gender", "string", "gender", "string"), ("age", "long", "age", "long"), ("tenure", "long", "tenure", "long"), ("balance", "double", "balance", "double"), ("numofproducts", "long", "numofproducts", "long"), ("hascrcard", "long", "hascrcard", "long"), ("isactivemember", "long", "isactivemember", "long"), ("estimatedsalary", "double", "estimatedsalary", "double"), ("exited", "long", "exited", "long"), ("file_type", "string", "file_type", "string")), caseSensitive = false, transformationContext = "applymapping1")
        val recordCount1 = applymapping1.count
        println("ds2 table record count: " + recordCount1)
        if(recordCount1 > 0) {
            spark_source_delta_df = applymapping1.toDF()
            println("ds2 table Schema: ")
            spark_source_delta_df.printSchema()
            EtlPipeline.run(sparkSession, glueJobId, propFilePath, module, spark_source_delta_df, "DS2")
        }
        
      } catch {
        case x: Exception => 
        {  
            println("Exception Dataframe is Empty") 
        } 
      }
    } else if (module.equals("TransformedToPublishedS3_ETLJob")) {
      try {
        
        val datasource0 = glueContext.getCatalogSource(database = "sa_transformed_db", tableName = "ds1", redshiftTmpDir = "", additionalOptions = JsonOptions("""{"mergeSchema": "true"}"""), transformationContext = "datasource0").getDynamicFrame()
        val recordCount0 = datasource0.count
        println("ds1 table record count: " + recordCount0)
        if(recordCount0 > 0) {
            spark_source_delta_df = datasource0.toDF()
            println("ds1 table Schema: ")
            spark_source_delta_df.printSchema()
            EtlPipeline.run(sparkSession, glueJobId, propFilePath, module, spark_source_delta_df, "DS1")
        }
        
        val datasource1 = glueContext.getCatalogSource(database = "sa_transformed_db", tableName = "ds2", redshiftTmpDir = "", additionalOptions = JsonOptions("""{"mergeSchema": "true"}"""), transformationContext = "datasource1").getDynamicFrame()
        val recordCount1 = datasource1.count
        println("ds2 table Delta record count: " + recordCount1)
        if(recordCount1 > 0) {
            spark_source_delta_df = datasource1.toDF()
            println("ds2 table Schema: ")
            spark_source_delta_df.printSchema()
            EtlPipeline.run(sparkSession, glueJobId, propFilePath, module, spark_source_delta_df, "DS2")
        }
        
      } catch {
        case x: Exception => 
        {  
            println("Exception Dataframe is Empty") 
        } 
      }
    }
    
    sparkSession.stop
    Job.commit()
  }
}
