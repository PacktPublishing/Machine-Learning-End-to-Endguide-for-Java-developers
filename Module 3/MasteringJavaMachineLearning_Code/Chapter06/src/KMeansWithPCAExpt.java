package org.mmlj.chapter9.expts;

import org.apache.spark.sql.SparkSession;

import java.util.ArrayList;
//$example on$
import java.util.Arrays;

import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.feature.PCA;
import org.apache.spark.ml.feature.PCAModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class KMeansWithPCAExpt {

	public static void main(String[] args) {
		SparkSession spark = SparkSession.builder()
				.master("local[8]")
				.appName("KMeansWithPCAExpt")
				.getOrCreate();

		// Load and parse data
		String filePath = "/home/kchoppella/book/Chapter09/data/covtypeNorm.csv";

		// Loads data.
		Dataset<Row> inDataset = spark.read()
				.format("com.databricks.spark.csv")
				.option("header", "true")
				.option("inferSchema", true)
				.load(filePath);
		ArrayList<String> inputColsList = new ArrayList<String>(Arrays.asList(inDataset.columns()));
		
		//Make single features column for feature vectors 
		inputColsList.remove("class");
		String[] inputCols = inputColsList.parallelStream().toArray(String[]::new);
		
		//Prepare dataset for training with all features in "features" column
		VectorAssembler assembler = new VectorAssembler().setInputCols(inputCols).setOutputCol("features");
		Dataset<Row> dataset = assembler.transform(inDataset);

		PCAModel pca = new PCA()
				.setK(16)
				.setInputCol("features")
				.setOutputCol("pcaFeatures")
				.fit(dataset);

		Dataset<Row> result = pca.transform(dataset).select("pcaFeatures");
		System.out.println("Explained variance:");
		System.out.println(pca.explainedVariance());
		result.show(false);
		
		KMeans kmeans = new KMeans().setK(27).setSeed(1L);
		KMeansModel model = kmeans.fit(dataset);

		// Evaluate clustering by computing Within Set Sum of Squared Errors.
		double WSSSE = model.computeCost(dataset);
		System.out.println("Within Set Sum of Squared Errors = " + WSSSE);

		// $example off$
		spark.stop();
	}
}