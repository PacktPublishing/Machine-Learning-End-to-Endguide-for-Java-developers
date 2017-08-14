package org.mmlj.chapter9.expts;

import java.io.BufferedWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.io.IOException;

import org.apache.spark.ml.feature.PCA;
import org.apache.spark.ml.feature.PCAModel;
import org.apache.spark.ml.feature.VectorAssembler;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import org.apache.spark.ml.clustering.GaussianMixture;
import org.apache.spark.ml.clustering.GaussianMixtureModel;

public class GaussianMixtureModelExpt {
	public static void main(String[] args) {

		SparkSession spark = SparkSession.builder()
				.master("local[8]")
				.appName("GaussianMixtureModelExpt")
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

		Dataset<Row> result = pca.transform(dataset).select("pcaFeatures").withColumnRenamed("pcaFeatures", "features");
		
		String outPath = "/home/kchoppella/book/Chapter09/data/gmm_params.csv";

		try {
			BufferedWriter writer = Files.newBufferedWriter(Paths.get(outPath));

			// Cluster the data into multiple classes using KMeans
			int numClusters = 27;
			GaussianMixtureModel gmm = new GaussianMixture()
					.setK(numClusters).
					fit(result);
			int numIterations = gmm.getK();
			// Output the parameters of the mixture model
			for (int i = 0; i < numIterations; i++) {
				String msg = String.format("Gaussian %d:\nweight=%f\nmu=%s\nsigma=\n%s\n\n",
					          i, 
					          gmm.weights()[i], 
					          gmm.gaussians()[i].mean(), 
					          gmm.gaussians()[i].cov());

				System.out.printf(msg);
				writer.write(msg + "\n");
				writer.flush();
			}
		} 
		catch (IOException iox) {
			System.out.println("Write Exception: \n");
			iox.printStackTrace();
		}
		finally {
		}
		
		spark.stop();
	}

}