package org.mmlj.chapter9.expts;

import org.apache.spark.sql.SparkSession;

import java.util.ArrayList;
//$example on$
import java.util.Arrays;
import org.apache.spark.ml.feature.PCA;
import org.apache.spark.ml.feature.PCAModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class PCAExpt {

	public static void main(String[] args) {
		SparkSession spark = SparkSession.builder()
				.master("local[8]")
				.appName("PCAExpt")
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
		// $example off$
		spark.stop();
	}
}