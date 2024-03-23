package main

import org.apache.spark.sql.functions.{avg, col, desc, lit, when}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}

object SparkDemo {
    def main(args: Array[String]): Unit = {
        val spark = SparkSession.builder()
            .master("local")
            .config("spark.master", "local")
            .appName("BigData")
            .getOrCreate();

        val csvProperties = Map("header" -> "true", "inferSchema" -> "true");

        val googlePlayStoreData = spark.read
            .options(csvProperties)
            .csv("src/main/resources/googleplaystore.csv");

        val userReviewsData = spark.read
            .options(csvProperties)
            .csv("src/main/resources/googleplaystore_user_reviews.csv")

        averageSentimentPolarityByApp(spark, userReviewsData);
        generateBestAppsCSV(spark, googlePlayStoreData, csvProperties);

        spark.stop();
    }

    /**
     * @param spark:SparkSession
     * @param userReviewsData:DataFrame
     *
     * @Exercise Part 1
     * @Objective From googleplaystore_user_reviews.csv create a Dataframe (df_1) with the following structure:
     *
     * |Column name	                 |  Data type  |  Default Value	        |  Notes                                                         |
     * |-----------------------------|-------------|------------------------|----------------------------------------------------------------|
     * |App	                         |  String     |  		                |                                                                |
     * |Average_Sentiment_Polarity   |  Double     |  0 (instead of NULL)	|  Average of the column Sentiment_Polarity grouped by App name  |
     *
     * @return df_1:DataFrame
     **/
    private def averageSentimentPolarityByApp(spark: SparkSession, userReviewsData: DataFrame): Unit = {
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

        val df_1 = spark.createDataFrame(avgUserReviewsDataByApp, schema);
        df_1.show();
    }

    /**
     * @param spark:SparkSession
     * @param userReviewsData:DataFrame
     *
     * @Exercise Part 2
     * @Objective Read googleplaystore.csv as a Dataframe and obtain all Apps with a "Rating" greater or equal to 4.0 sorted in descending order.
     *
     * @return Save that Dataframe as a CSV (delimiter: "§") named "best_apps.csv"
     *         df_2:DataFrame
     **/
    private def generateBestAppsCSV(spark: SparkSession, googlePlayStoreData: DataFrame, csvProperties: Map[String, String]): Unit = {
        googlePlayStoreData
            .withColumn("Rating", col("Rating").cast("double"))
            .filter(!col("Rating").isNaN and col("Rating") >= 4.0)
            .orderBy(desc("Rating"))
            .write
            .option("header", "true")
            .option("delimiter", "§")
            .mode(SaveMode.Overwrite)
            .csv("src/main/resources/best_apps.csv");

        val df_2 = spark.read
            .options(csvProperties)
            .option("delimiter", "§")
            .csv("src/main/resources/best_apps.csv");
        df_2.show();
    }
}