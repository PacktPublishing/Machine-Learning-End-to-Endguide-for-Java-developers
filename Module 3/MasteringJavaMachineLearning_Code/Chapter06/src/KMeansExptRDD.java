package org.mmlj.chapter9.expts;

import java.io.BufferedWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.io.IOException;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

public class KMeansExptRDD {
	public static void main(String[] args) {

		SparkConf sparkConf = new SparkConf().setAppName("KMeansExpt").setMaster("local[16]")
				.set("spark.executor.memory", "1g");
		JavaSparkContext jsc = new JavaSparkContext(sparkConf);

		// Load and parse data
		String path = "/home/kchoppella/book/Chapter09/data/covtypeNorm.csv";

		JavaRDD<String> cdata = jsc.textFile(path);
		JavaRDD<Vector> parsedData = cdata.map(s -> {
			String[] sarray = s.split(",");
			double[] features = new double[sarray.length - 1];
			for (int i = 0; i < sarray.length - 1; i++) {
				features[i] = Double.parseDouble(sarray[i]);
			};
			return Vectors.dense(features);
		});
		parsedData.cache();

		JavaRDD<LabeledPoint> pData = cdata.map(s -> {
			String[] sarray = s.split(",");
			double[] features = new double[sarray.length - 1];
			for (int i = 0; i < sarray.length - 1; i++) {
				features[i] = Double.parseDouble(sarray[i]);
			}
			;
			Double label = new Double(Double.parseDouble(sarray[sarray.length - 1]));
			return new LabeledPoint(label, Vectors.dense(features));
		});

		String outPath = "/home/kchoppella/book/Chapter09/data/kmeans_cost.csv";
		try {
			BufferedWriter writer = Files.newBufferedWriter(Paths.get(outPath));

			// Cluster the data into two classes using KMeans
			int numClusters = 20;
			int numIterations = 10;
			for (int k = numClusters - 1; k < numClusters + 1; k += 1) {
				KMeansModel clusters = KMeans.train(pData.map(f -> f.features()).rdd(), k, numIterations);

				// Evaluate clustering by computing Within Set Sum of Squared
				// Errors
				double WSSSE = clusters.computeCost(parsedData.rdd());
				String msg = (k + "," + WSSSE).toString();
				System.out.println(msg);
				writer.write(msg + "\n");
				writer.flush();
			}
		} catch (IOException iox) {
			System.out.println("Write Exception: \n");
			iox.printStackTrace();
		} finally {
		}

		jsc.close();
		jsc.stop();
	}
}