package org.mmlj.chapter9.expts;

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.VectorIndexer;
import org.apache.spark.ml.feature.VectorIndexerModel;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.clustering.GaussianMixture;
import org.apache.spark.ml.clustering.GaussianMixtureModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;

public class RandomForestExpt {
	public static void main(String[] args) {

		SparkSession spark = SparkSession.builder()
				.master("local[8]")
				.appName("RandomForestExpt")
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

		// Index labels, adding metadata to the label column.
		// Fit on whole dataset to include all labels in index.
		StringIndexerModel labelIndexer = new StringIndexer()
		  .setInputCol("class")
		  .setOutputCol("indexedLabel")
		  .fit(dataset);
		// Automatically identify categorical features, and index them.
		// Set maxCategories so features with > 4 distinct values are treated as continuous.
		VectorIndexerModel featureIndexer = new VectorIndexer()
		  .setInputCol("features")
		  .setOutputCol("indexedFeatures")
		  .setMaxCategories(2)
		  .fit(dataset);

		// Split the data into training and test sets (30% held out for testing)
		Dataset<Row>[] splits = dataset.randomSplit(new double[] {0.7, 0.3});
		Dataset<Row> trainingData = splits[0];
		Dataset<Row> testData = splits[1];

		// Train a GBT model.
		RandomForestClassifier rf = new RandomForestClassifier()
		  .setLabelCol("indexedLabel")
		  .setFeaturesCol("indexedFeatures")
		  .setImpurity("gini")
		  .setMaxDepth(5)
		  .setNumTrees(20)
		  .setSeed(1234);

		// Convert indexed labels back to original labels.
		IndexToString labelConverter = new IndexToString()
		  .setInputCol("prediction")
		  .setOutputCol("predictedLabel")
		  .setLabels(labelIndexer.labels());

		// Chain indexers and GBT in a Pipeline.
		Pipeline pipeline = new Pipeline()
		  .setStages(new PipelineStage[] {labelIndexer, featureIndexer, rf, labelConverter});

		// Train model. This also runs the indexers.
		PipelineModel model = pipeline.fit(trainingData);

		// Make predictions.
		Dataset<Row> predictions = model.transform(testData);

		// Select example rows to display.
		predictions.select("predictedLabel", "class", "features").show(5);

		// Select (prediction, true label) and compute test error.
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
		  .setLabelCol("indexedLabel")
		  .setPredictionCol("prediction");
		
		String outPath = "/home/kchoppella/book/Chapter09/data/randomforest_evaluation.csv";
		
		try {
			BufferedWriter writer = Files.newBufferedWriter(Paths.get(outPath));

			evaluator.setMetricName("accuracy");
			double accuracy = evaluator.evaluate(predictions);
			
			evaluator.setMetricName("f1");
			double f1 = evaluator.evaluate(predictions);
			
			evaluator.setMetricName("weightedPrecision");
			double weightedPrecision = evaluator.evaluate(predictions);
		
			evaluator.setMetricName("weightedRecall");
			double weightedRecall = evaluator.evaluate(predictions);
			String msg = String.format("%f,%f,%f,%f\n",
					          accuracy, 
					          f1, 
					          weightedPrecision,
					          weightedRecall);
			String header = "Accuracy,F1 measure,Weighted precision,Weighted recall";
			System.out.println(header);
			System.out.printf(msg);
			writer.write(header + "\n");
			writer.write(msg + "\n");
			writer.flush();
			
/*			RandomForestClassificationModel rfModel = (RandomForestClassificationModel)(model.stages()[2]);
			System.out.println("Learned classification RandomForest model:\n" + rfModel.toDebugString());
*/		} 
		catch (IOException iox) {
			System.out.println("Write Exception: \n");
			iox.printStackTrace();
		}
		finally {
		}

		spark.stop();
	}
}