package tiago.cebola

import org.apache.spark.SparkConf
import org.apache.spark.sql.functions.{avg, col, collect_list, max, count, desc, explode, lit, regexp_extract, round, split, to_date, when}
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}

import scala.math.Ordering.Implicits.infixOrderingOps

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

    /**
     * @Objective Init spark local application with master configs and only with ERROR logs.
     *
     * @return spark:SparkSession
     */
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
        userReviewsData
            .groupBy("App")
            .agg(avg("Sentiment_Polarity")
                .cast("double")
                .as("Average_Sentiment_Polarity"))
            .na.fill(0)
            .select("App", "Average_Sentiment_Polarity")
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
            .filter(col("Rating").isNotNull && !col("Rating").isNaN && col("Rating") >= 4.0)
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
        googlePlayStoreData
            .withColumn("Rating", when(!col("Rating").isNaN, col("Rating").cast("double"))
                .otherwise(lit(null)))
            .withColumn("Reviews", col("Reviews").cast("long")).na.fill(0)
            .withColumn("Size", when(col("Size").endsWith("M"), regexp_extract(col("Size"), "^(\\d+(?:\\.\\d+)?)M", 1).cast("double"))
                .otherwise(when(col("Size").endsWith("K"), regexp_extract(col("Size"), "^(\\d+(?:\\.\\d+)?)K", 1).cast("double") / 1024.0)
                    .otherwise(lit(null))))
            .withColumn("Price", when(col("Price").contains("$"), round(regexp_extract(col("Price"), "^\\$(\\d+(\\.\\d+)?)", 1).cast("double") * lit(0.9), 2))
                .otherwise(when(col("Price") === 0, col("Price").cast("double"))
                    .otherwise(lit(null))))
            .withColumn("Last_Updated", to_date(col("Last Updated"), "MMMM d, yyyy"))
            .withColumn("Genres", split(col("Genres"), ";").cast("array<string>"))
            .groupBy("App")
            .agg(
                collect_list("Category").as("Categories"),
                max("Rating").as("Rating"),
                max("Reviews").as("Reviews"),
                max("Size").as("Size"),
                max("Installs").as("Installs"),
                max("Type").as("Type"),
                max("Price").as("Price"),
                max("Content Rating").as("Content_Rating"),
                max("Genres").as("Genres"),
                max("Last_Updated").as("Last_Updated"),
                max("Current Ver").as("Current_Version"),
                max("Android Ver").as("Minimum_Android_Version"))
            .dropDuplicates("App")
            .select("App", "Categories", "Rating", "Reviews", "Size", "Installs", "Type", "Price", "Content_Rating", "Genres", "Last_Updated", "Current_Version", "Minimum_Android_Version");
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
        df_3.join(df_1, df_1.col("App") === df_3.col("App"), "left")
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
        df_4.withColumn("Genre", explode(col("Genres")))
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