using System;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Calibration;
using Microsoft.ML.Transforms;

namespace TitanicML
{
    class Program
    {
        /* I got a 0.74641 Score on Kaggle with following setup, far from optimal and
        is something i will be playing with as i learn more regarding machine learning */
        static readonly string _trainingData = Path.Combine(Environment.CurrentDirectory, "Data", "train.csv");
        static readonly string _testData = Path.Combine(Environment.CurrentDirectory, "Data", "test.csv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static void Main(string[] args)
        {
            // Start the program.
            TrainModel();
        }

        private static void TrainModel()
        {
            // Setup the Machine Learning Context
            var mlContext = new MLContext();

            // Setup the Training data by reading it from a CSV file and build up a IDataView based on the Passenger Model.
            var trainingData = mlContext.Data.ReadFromTextFile<Passenger>(_trainingData, hasHeader: true, separatorChar: ',');

            // Do Transformation on the data, including dropping columns that doesn't have any value to the predictions.
            var dataPipeline = mlContext.Transforms
                // Dropping Passenger Id's, Names, Ticket's, Fare's and Cabin.
                .DropColumns(nameof(Passenger.PassengerId), nameof(Passenger.Name), nameof(Passenger.Ticket), nameof(Passenger.Fare), nameof(Passenger.Cabin))
                // Replacing missing values on Age with a Mean Value based on Age.
                .Append(mlContext.Transforms.ReplaceMissingValues(nameof(Passenger.Age), nameof(Passenger.Age), MissingValueReplacingTransformer.ColumnInfo.ReplacementMode.Mean))
                // Converts Gender, Embark and Passenger Class from a Text Column to a one-hot encoded vector.
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(nameof(Passenger.Gender), nameof(Passenger.Gender)))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(nameof(Passenger.Embarked), nameof(Passenger.Embarked)))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(nameof(Passenger.PassengerClass), nameof(Passenger.PassengerClass)))
                // Concates Passenger Class, Gender, Age, SiblingsOrSpouses, ParentsOrChildren and Embarked into Features Column.
                .Append(mlContext.Transforms.Concatenate("Features", nameof(Passenger.PassengerClass), nameof(Passenger.Gender),
                    nameof(Passenger.Age), nameof(Passenger.SiblingsOrSpouses),
                    nameof(Passenger.ParentsOrChildren), nameof(Passenger.Embarked)))
                // Setup which trainer to use for training, Used FastTree but there are other options.
                .Append(mlContext.BinaryClassification.Trainers.FastTree(nameof(Passenger.Survived)))
                // Fit the Training Data.
                .Fit(trainingData);

            // Time to do a Evaluation of the Training Data.
            EvaluateModel(mlContext, dataPipeline);

            Console.WriteLine();

            // Time to do a Prediction on the Test Data.
            PredictTestData(mlContext, dataPipeline);

            Console.WriteLine();

            // Currently not being used since not required for Kaggle.com competition.
            //SaveModel(mlContext, dataPipeline);
        }

    private static void EvaluateModel(MLContext mlContext, ITransformer dataPipeline)
    {
        // Once again, Open the training data into a IDataView
        var trainingData = mlContext.Data.ReadFromTextFile<Passenger>(_trainingData, hasHeader: true, separatorChar: ',');

        // Run a BinaryClassification of the Fitted Training Data on the newly opened Training Data and output the Accuracy / F1 of the Classification.
        var statistics = mlContext.BinaryClassification.EvaluateNonCalibrated(dataPipeline.Transform(trainingData), nameof(OutputModel.Survived));

        // Write the Output to the Console.
        Console.WriteLine("Training Performance: ");
        Console.WriteLine($"\tAccuracy: {statistics.Accuracy}");
        Console.WriteLine($"\tF1: {statistics.F1Score}");
    }

    private static void PredictTestData(MLContext mlContext, ITransformer dataPipeline)
    {
        // Create a Predicition Engine using the Predicted Data Model and Output Model.
        var predictor = dataPipeline.CreatePredictionEngine<PredictedData, OutputModel>(mlContext);

        // Open the Test Data and assign it to a IDataView based on Predicted Data Model.
        var evalData = mlContext.Data.ReadFromTextFile<PredictedData>(_testData, hasHeader: true, separatorChar: ',');

        // Setup a string builder.
        var csvWriter = new StringBuilder();

        // Add PassengerId,Surivived to the first line of the string and end with a \n
        csvWriter.Append("PassengerId,Survived\n");
        // Write out PassengerId, Survived to Console (Not required at this point if you want to submit the predicition to Kaggle.Com)
        Console.WriteLine($"PassengerId, Survived");

        // Setup a loop of the evaluated data.
        foreach(var row in evalData.Preview(2000).RowView)
        {
            // Create a new Predicted Data model per loop
            var inputModel = new PredictedData
            {
                Embarked = row.Values[10].Value.ToString(),
                PassengerClass = (float)row.Values[1].Value,
                Gender = row.Values[3].Value.ToString(),
                Age = (float)row.Values[4].Value,
                ParentsOrChildren = (float)row.Values[6].Value,
                SiblingsOrSpouses = (float)row.Values[5].Value,
                Cabin = row.Values[9].Value.ToString(),
                Name = row.Values[2].Value.ToString(),
                Fare = (double)row.Values[8].Value,
                Ticket = row.Values[7].Value.ToString(),
                PassengerId = (int)row.Values[0].Value
            };

            // Do the Prediction on the created InputModel
            var prediction = predictor.Predict(inputModel);

            // Write out the PassengerId and Survived as 1 (Survived) or 0 (Didn't Survive) to Console, Also not needed for submission to Kaggle.com
            Console.WriteLine($"{inputModel.PassengerId}, {(prediction.Survived ? "1" : "0")}");

            // Setup two variables and create a new line that ends with \n and append it to the stringbuilder.
            var pId = inputModel.PassengerId;
            var _survived = prediction.Survived ? "1" : "0";
            var newLine = $"{pId},{_survived}\n";
            csvWriter.Append(newLine);
        }

        // Save the Predicition data in the format that Kaggle.com wanted it.
        File.WriteAllText("Data/submission.csv", csvWriter.ToString());
    }

    private static void SaveModel(MLContext mlContext, ITransformer dataPipeline)
    {
        // Saves the Trained Model into a Model.zip file.
        Console.WriteLine("Saving the model to Model.zip");
        using(var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
        {
            dataPipeline.SaveTo(mlContext, fs);
        }
    }
  }
}