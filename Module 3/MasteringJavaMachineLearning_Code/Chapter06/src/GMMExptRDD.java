package org.mmlj.chapter9.expts;

import java.io.BufferedWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.io.IOException;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

// $example on$
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;

import scala.Tuple2;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.mllib.clustering.GaussianMixture;
import org.apache.spark.mllib.clustering.GaussianMixtureModel;
// $example off$

public class GMMExptRDD {
	public static void main(String[] args) {

		// SparkConf conf = new SparkConf().setAppName("KMeansExpt");
		SparkConf sparkConf = new SparkConf().setAppName("KMeansExpt").setMaster("local[2]")
				.set("spark.executor.memory", "1g");
		JavaSparkContext jsc = new JavaSparkContext(sparkConf);

		// $example on$
		// Load and parse data
		String path = "/home/kchoppella/book/Chapter09/data/covtypeNorm.csv";

		// JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(jsc.sc(),
		// path).toJavaRDD();
		// Split the data into training and test sets (30% held out for testing)

		JavaRDD<String> cdata = jsc.textFile(path);
		JavaRDD<Vector> parsedData = cdata.map(s -> {
			String[] sarray = s.split(",");
			double[] features = new double[sarray.length - 1];
			for (int i = 0; i < sarray.length - 1; i++) {
				features[i] = Double.parseDouble(sarray[i]);
			}
			;
			// Double label = new Double
			// (Double.parseDouble(sarray[sarray.length -1 ]));
			// return new LabeledPoint(label, Vectors.dense(features));
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

		JavaRDD<LabeledPoint>[] splits = pData.randomSplit(new double[] { 0.7, 0.3 });
		JavaRDD<LabeledPoint> trainingData = splits[0];
		JavaRDD<LabeledPoint> testData = splits[1];

		String outPath = "/home/kchoppella/book/Chapter09/data/gmm_params.csv";
		try {
			BufferedWriter writer = Files.newBufferedWriter(Paths.get(outPath));

			// Cluster the data into multiple classes using KMeans
			int numClusters = 30;
			GaussianMixtureModel gmm = new GaussianMixture().setK(numClusters).run(parsedData.rdd());
			int numIterations = gmm.k();
			// Output the parameters of the mixture model
			for (int j = 0; j < numIterations; j++) {
				String msg = String.format("weight=%f\nmu=%s\nsigma=\n%s\n", gmm.weights()[j], gmm.gaussians()[j].mu(),
						gmm.gaussians()[j].sigma());
				System.out.printf(msg);
				writer.write(msg + "\n");
				writer.flush();
			}
		} catch (IOException iox) {
			System.out.println("Write Exception: \n");
			iox.printStackTrace();
		} finally {
		}

		jsc.stop();
	}

}