package org.mmlj.chapter9.expts;

import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.clustering.BisectingKMeans;
import org.apache.spark.ml.clustering.BisectingKMeansModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import java.util.ArrayList;
import java.util.Arrays;

public class BisectingKMeansExpt {
	public static void main(String[] args) {

		SparkSession spark = SparkSession.builder()
				.master("local[8]")
				.appName("BisectingKMeansExpt")
				.getOrCreate();
 
		// Load and parse data
		String filePath = "/home/kchoppella/book/Chapter09/data/covtypeNorm.csv";

		// Loads data.
		Dataset<Row> inDataset = spark.read()
				.format("com.databricks.spark.csv")
				.option("header", "true")
				.option("inferSchema", true)
				.load(filePath);
		
		//Make single features column for feature vectors 
		ArrayList<String> inputColsList = new ArrayList<String>(Arrays.asList(inDataset.columns()));
		inputColsList.remove("class");
		String[] inputCols = inputColsList.parallelStream().toArray(String[]::new);
		
		//Prepare dataset for training with all features in "features" column
		VectorAssembler assembler = new VectorAssembler().setInputCols(inputCols).setOutputCol("features");
		Dataset<Row> dataset = assembler.transform(inDataset);

		// Trains a bisecting k-means model.
		BisectingKMeans bkm = new BisectingKMeans().setK(27).setSeed(1);
		BisectingKMeansModel model = bkm.fit(dataset);

		// Evaluate clustering by computing Within Set Sum of Squared Errors.
		double WSSSE = model.computeCost(dataset);
		System.out.println("Within Set Sum of Squared Errors = " + WSSSE);

		// Shows the result.
		Vector[] centers = model.clusterCenters();
		System.out.println("Cluster Centers: ");
		for (Vector center: centers) {
		  System.out.println(center);
		}
		
		spark.stop();
	}
}