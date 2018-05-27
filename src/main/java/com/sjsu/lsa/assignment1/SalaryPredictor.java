package com.sjsu.lsa.assignment1;

import static org.apache.spark.sql.functions.avg;
import static org.apache.spark.sql.functions.col;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import com.sjsu.utils.SparkConnection;
import org.apache.spark.api.java.JavaRDD;

public class SalaryPredictor {

	public static void main(String[] args) {
		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);
		JavaSparkContext spContext = SparkConnection.getContext();
		SparkSession spSession = SparkConnection.getSession();
		
		
		Dataset<Row> autoDf = spSession.read()
				.option("header","true")
				.csv("data/train.csv");
		autoDf.show(5);
		autoDf.printSchema();
		
		
		System.out.println("==========================================");
		Dataset<Row> summaryData = autoDf.groupBy(col("union_code"))
				.agg(avg(autoDf.col("salary")));
			summaryData.show(30000);
		
		/**autoDf.select(col("department_code"), col("salary")).filter(col("salary").$less("1")).distinct().show(60);
			
			*/
		
		StructType autoSchema = DataTypes
				.createStructType(new StructField[] {
						DataTypes.createStructField("id", DataTypes.DoubleType, false),
						DataTypes.createStructField("worker_group_code", DataTypes.StringType, false),
						DataTypes.createStructField("worker_group_name", DataTypes.StringType, false),
						DataTypes.createStructField("department_code", DataTypes.StringType, false),
						DataTypes.createStructField("department_name", DataTypes.StringType, false),
						DataTypes.createStructField("union_code", DataTypes.DoubleType, false),
						DataTypes.createStructField("union_name", DataTypes.StringType, false),
						DataTypes.createStructField("job_group_code", DataTypes.StringType, false),
						DataTypes.createStructField("job_group",DataTypes.StringType, false),
						DataTypes.createStructField("job_code",DataTypes.StringType, false),
						DataTypes.createStructField("job",DataTypes.StringType, false),
						DataTypes.createStructField("salary",DataTypes.DoubleType, false),
						DataTypes.createStructField("mean_job_id",DataTypes.DoubleType, false),
						DataTypes.createStructField("mean_dept_id",DataTypes.DoubleType, false),
						DataTypes.createStructField("worker_mean_id",DataTypes.DoubleType, false)
					});
		
		JavaRDD<Row> rdd1 = autoDf.toJavaRDD().repartition(2);
		
		final Broadcast<Double> avgHP = spContext.broadcast(0.0);
		
		JavaRDD<Row> rdd2 = rdd1.map( new Function<Row, Row>() {

			public Row call(Row iRow) throws Exception {
				
				double hp = (iRow.getString(5).equals("") ?
						avgHP.value() : Double.valueOf(iRow.getString(5))); 
				
				Row retRow = RowFactory.create( Double.valueOf(iRow.getString(0)), 
								iRow.getString(1),
								iRow.getString(2),
								iRow.getString(3),
								iRow.getString(4),
								Double.valueOf(hp),
								iRow.getString(6),
								iRow.getString(7),
								iRow.getString(8),
								iRow.getString(9),
								iRow.getString(10),
								Double.valueOf(iRow.getString(11)),
								Double.valueOf(iRow.getString(12)),
								Double.valueOf(iRow.getString(13)),
								Double.valueOf(iRow.getString(14))
						);
				
				return retRow;
			}

		});
		
		
		Dataset<Row> autoCleansedDf = spSession.createDataFrame(rdd2, autoSchema);
		autoCleansedDf.printSchema();		
		for ( StructField field : autoSchema.fields() ) {
			if ( ! field.dataType().equals(DataTypes.StringType)) {
				System.out.println( "Correlation between Salary and " + field.name()
				 	+ " = " + autoCleansedDf.stat().corr("salary", field.name()) );
			}
		}
		
		/*--------------------------------------------------------------------------
		Prepare for Machine Learning. 
		--------------------------------------------------------------------------*/
		
		//Convert data to labeled Point structure
		JavaRDD<Row> rdd3 = autoCleansedDf.toJavaRDD().repartition(2);
		
		JavaRDD<LabeledPoint> rdd4 = rdd3.map( new Function<Row, LabeledPoint>() {

			public LabeledPoint call(Row iRow) throws Exception {
				
				LabeledPoint lp = new LabeledPoint(iRow.getDouble(11) , 
									Vectors.dense(iRow.getDouble(12)));
				
				return lp;
			}

		});

		Dataset<Row> autoLp = spSession.createDataFrame(rdd4, LabeledPoint.class);
		autoLp.show(5);
		
		// Split the data into training and test sets (10% held out for testing).
		Dataset<Row>[] splits = autoLp.randomSplit(new double[]{0.9, 0.1});
		Dataset<Row> trainingData = splits[0];
		Dataset<Row> testData = splits[1];
		
		/*--------------------------------------------------------------------------
		Perform machine learning. 
		--------------------------------------------------------------------------*/	
		
		//Create the object
		LinearRegression lr = new LinearRegression();
		//Create the model
		LinearRegressionModel lrModel = lr.fit(trainingData);
		
		//Print out coefficients and intercept for LR
		System.out.println("Coefficients: "
				  + lrModel.coefficients() + " Intercept: " + lrModel.intercept());
		
		//Predict on test data
		Dataset<Row> predictions = lrModel.transform(testData);
		
		//View results
		predictions.select("label", "prediction", "features").show(5);
		
		//Compute R2 for the model on test data.
		RegressionEvaluator evaluator = new RegressionEvaluator()
				  .setLabelCol("label")
				  .setPredictionCol("prediction")
				  .setMetricName("r2");
		double r2 = evaluator.evaluate(predictions);
		System.out.println("R2 on test data = " + r2);
		
	}

}
