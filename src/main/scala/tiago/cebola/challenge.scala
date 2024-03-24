package tiago.cebola

import org.apache.spark.SparkConf
import org.apache.spark.sql.functions.{avg, col, collect_list, concat_ws, count, desc, explode, lit, regexp_extract, round, split, to_date, when}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}

object challenge {
    private final val csvReadingProperties = Map("header" -> "true", "inferSchema" -> "true");
    private final val csvWritingProperties = Map("header" -> "true", "delimiter" -> "§");

    def main(args: Array[String]): Unit = {
        val spark = initSpark();

        val googlePlayStoreData = spark.read
            .options(csvReadingProperties)
            .csv("src/main/resources/originals/googleplaystore.csv");

        val userReviewsData = spark.read
            .options(csvReadingProperties)
            .csv("src/main/resources/originals/googleplaystore_user_reviews.csv");

        println("\nPart 1 - Average Sentiment Polarity By App");
        val df_1 = averageSentimentPolarityByApp(spark, userReviewsData);
        df_1.show();
        df_1.printSchema();

        println("\nPart 2 - Generated Best Apps CSV");
        val df_2 = generateBestAppsCSV(spark, googlePlayStoreData);
        df_2.show();

        println("\nPart 3 - Google Play Store Data Grouped By App And Standardize")
        val df_3 = groupByAppAndStandardize(googlePlayStoreData);
        df_3.show();
        df_3.printSchema();

        println("\nPart 4 - Generated Cleaned Google Play Store Data GZIP");
        val df_4 = cleanGooglePlayStoreData(spark, df_1, df_3);
        df_4.show();

        println("\nPart 5 - Generated Google Play Store Metrics By Genre GZIP");
        val df_5 = getGooglePlayStoreMetricsByGenre(spark, df_4);
        df_5.show();

        spark.stop();
    }

    private def initSpark(): SparkSession = {
        val sparkConfig = new SparkConf()
            .setAppName("BigData-Challenge")
            .set("spark.master", "local");

        val sparkSession = SparkSession.builder()
            .master("local")
            .config(sparkConfig)
            .getOrCreate();

        sparkSession.sparkContext.setLogLevel("ERROR");
        sparkSession;
    }

    /**
     * @param spark:SparkSession
     * @param userReviewsData:DataFrame
     * @Challenge Part 1
     * @Objective From googleplaystore_user_reviews.csv create a Dataframe (df_1) with the following structure:
     *
     * |Column name	                 |  Data type  |  Default Value	        |  Notes                                                         |
     * |-----------------------------|-------------|------------------------|----------------------------------------------------------------|
     * |App	                         |  String     |  		                |                                                                |
     * |Average_Sentiment_Polarity   |  Double     |  0 (instead of NULL)	|  Average of the column Sentiment_Polarity grouped by App name  |
     * @return df_1:DataFrame
     **/
    private def averageSentimentPolarityByApp(spark: SparkSession, userReviewsData: DataFrame): DataFrame = {
        val avgUserReviewsDataByApp = userReviewsData
            .groupBy("App")
            .agg(avg("Sentiment_Polarity").cast("double").as("Average_Sentiment_Polarity"))
            .withColumn("Average_Sentiment_Polarity",
                when(col("Average_Sentiment_Polarity").isNaN, lit(0.0))
                    .otherwise(col("Average_Sentiment_Polarity")))
            .select("App", "Average_Sentiment_Polarity")
            .collectAsList();

        val schema = new StructType()
            .add(StructField("App", StringType, nullable = true))
            .add(StructField("Average_Sentiment_Polarity", DoubleType, nullable = false));

        spark.createDataFrame(avgUserReviewsDataByApp, schema);
    }

    /**
     * @param spark:SparkSession
     * @param googlePlayStoreData:DataFrame
     *
     * @Challenge Part 2
     * @Objective Read googleplaystore.csv as a Dataframe and obtain all Apps with a "Rating" greater or equal to 4.0 sorted in descending order.
     *
     * @return Save that Dataframe as a CSV (delimiter: "§") named "best_apps.csv"
     *         df_2:DataFrame
     **/
    private def generateBestAppsCSV(spark: SparkSession, googlePlayStoreData: DataFrame): DataFrame = {
        googlePlayStoreData
            .withColumn("Rating", col("Rating").cast("double"))
            .filter(!col("Rating").isNaN and col("Rating") >= 4.0)
            .orderBy(desc("Rating"))
            .write
            .options(csvWritingProperties)
            .mode(SaveMode.Overwrite)
            .csv("src/main/resources/best_apps.csv");

        spark.read
            .options(csvReadingProperties)
            .option("delimiter", "§")
            .csv("src/main/resources/best_apps.csv");
    }

    /**
     * @param googlePlayStoreData:DataFrame
     *
     * @Challenge Part 3
     * @Objective From googleplaystore.csv create a Dataframe (df_3) with the structure from the table below:
     *
     * |Column name	             |  Data type      |  Default Value  |  IMPORTANT NOTES                                                                                                                |
     * |-------------------------|-----------------|-----------------|---------------------------------------------------------------------------------------------------------------------------------|
     * |App	                     |  String         |  		         |  Remove duplicates                                                                                                              |
     * |Categories               |  Array[String]  |  	             |  Rename column                                                                                                                  |
     * |Rating                   |  Double         |  null	         |                                                                                                                                 |
     * |Reviews                  |  Long           |  0	             |                                                                                                                                 |
     * |Size                     |  Double         |  null	         |  Convert from string to double (value in MB). Attention - Not all values end in "M"                                             |
     * |Installs                 |  String         |  null	         |                                                                                                                                 |
     * |Type                     |  String         |  null	         |                                                                                                                                 |
     * |Price                    |  Double         |  null	         |  Convert from string to double and present the value in euros (All values are in dollars, consider conversion rate: 1$ = 0.9€)  |
     * |Content_Rating           |  String         |  null	         |  Rename column from 'Content Rating'                                                                                            |
     * |Genres                   |  Array[String]  |  null	         |  Convert string to array of strings (delimiter: ";")                                                                            |
     * |Last_Updated             |  Date           |  null	         |  Convert string to date. Rename column from 'Last Updated'                                                                      |
     * |Current_Version          |  String         |  null	         |  Rename column from 'Current Ver'                                                                                               |
     * |Minimum_Android_Version  |  String         |  null	         |  Rename column from 'Android Ver'                                                                                               |
     *
     * @return df_3:DataFrame
     **/
    private def groupByAppAndStandardize(googlePlayStoreData: DataFrame): DataFrame = {
        val df_3 = googlePlayStoreData
            .withColumn("Current_Version", col("Current Ver"))
            .withColumn("Minimum_Android_Version", col("Android Ver"))
            .withColumn("Content_Rating", col("Content Rating"))
            .withColumn("Last_Updated", to_date(col("Last Updated"), "MMMM d, yyyy"))
            .withColumn("Rating", col("Rating").cast("double"))
            .withColumn("Reviews", col("Reviews").cast("long"))
            .withColumn("Price", when(col("Price").contains("$"), round(regexp_extract(col("Price"), "^\\$(\\d+(\\.\\d+)?)", 1).cast("double") * lit(0.9), 2))
                .otherwise(lit(0.0)))
            .withColumn("Size", when(col("Size").endsWith("M") , regexp_extract(col("Size"), "^(\\d+(?:\\.\\d+)?)M", 1).cast("double"))
                .otherwise(lit(null)))
            .groupBy("App", "Rating", "Reviews", "Size", "Installs", "Type", "Price", "Content_Rating", "Last_Updated", "Current_Version", "Minimum_Android_Version")
            .agg(
                collect_list("Category").as("Categories"),
                collect_list(concat_ws(";", col("Genres"))).as("Genres"))
            .select("App", "Categories", "Rating", "Reviews", "Size", "Installs", "Type", "Price", "Content_Rating", "Genres", "Last_Updated", "Current_Version", "Minimum_Android_Version");

        df_3;
    }

    /**
     * @param spark:SparkSession
     * @param df_1:DataFrame
     * @param df_3:DataFrame
     *
     * @Challenge Part 4
     * @Objective Given the Dataframes produced by Part 1 and 3, produce a Dataframe with all its information plus its 'Average_Sentiment_Polarity' calculated in Exercise 1
     *            Save the final Dataframe as a parquet file with gzip compression with the name "googleplaystore_cleaned"
     *
     *
     * @return Save that Dataframe as a Parquet with gzip (delimiter: "§") named "googleplaystore_cleaned.gz"
     *         df_4:DataFrame
     **/
    private def cleanGooglePlayStoreData(spark: SparkSession, df_1: DataFrame, df_3: DataFrame): DataFrame = {
        df_3
            .join(df_1, df_1.col("App") === df_3.col("App"), "inner")
            .drop(df_1.col("App"))
            .write
            .options(csvWritingProperties)
            .option("compression", "gzip")
            .mode(SaveMode.Overwrite)
            .parquet("src/main/resources/googleplaystore_cleaned.gz");

        spark.read
            .options(csvReadingProperties)
            .option("delimiter", "§")
            .option("compression", "gzip")
            .parquet("src/main/resources/googleplaystore_cleaned.gz");
    }

    /**
     * @param spark:SparkSession
     * @param googlePlayStoreData:DataFrame
     *
     * @Challenge Part 5
     * @Objective Using df_4 create a new Dataframe (df_5) containing the number of applications, the average rating
     *            and the average sentiment polarity by genre and save it as a parquet file with gzip compression with the name "googleplaystore_metrics".
     *
     * @Notes I decided to use df_4 instead of df_3 due to the need to have the "Average_Sentiment_Polarity" column
     *
     * @return Save that Dataframe as a Parquet with gzip (delimiter: "§") named "googleplaystore_metrics.gz"
     *         df_5:DataFrame
     **/
    private def getGooglePlayStoreMetricsByGenre(spark: SparkSession, df_4: DataFrame): DataFrame = {
        df_4
            .withColumn("Genre", explode(col("Genres")))
            .withColumn("Genre", explode(split(col("Genre"), ";")))
            .groupBy("Genre")
            .agg(
                count("*").as("Count"),
                avg("Rating").as("Average_Rating"),
                avg("Average_Sentiment_Polarity").as("Average_Sentiment_Polarity"))
            .write
            .options(csvWritingProperties)
            .option("compression", "gzip")
            .mode(SaveMode.Overwrite)
            .parquet("src/main/resources/googleplaystore_metrics.gz");

        spark.read
            .options(csvReadingProperties)
            .option("delimiter", "§")
            .option("compression", "gzip")
            .parquet("src/main/resources/googleplaystore_metrics.gz");
    }
}