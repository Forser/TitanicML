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
        static readonly string _trainingData = Path.Combine(Environment.CurrentDirectory, "Data", "train.csv");
        static readonly string _testData = Path.Combine(Environment.CurrentDirectory, "Data", "test.csv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        static void Main(string[] args)
        {
            TrainModel();
        }

        private static void TrainModel()
        {
            var mlContext = new MLContext();

            var trainingData = mlContext.Data.ReadFromTextFile<Passenger>(_trainingData, hasHeader: true, separatorChar: ',');

            var dataPipeline = mlContext.Transforms
                .DropColumns(nameof(Passenger.PassengerId), nameof(Passenger.Name), nameof(Passenger.Ticket), nameof(Passenger.Fare), nameof(Passenger.Cabin))
                .Append(mlContext.Transforms.ReplaceMissingValues(nameof(Passenger.Age), nameof(Passenger.Age), MissingValueReplacingTransformer.ColumnInfo.ReplacementMode.Mean))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(nameof(Passenger.Gender), nameof(Passenger.Gender)))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(nameof(Passenger.Embarked), nameof(Passenger.Embarked)))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(nameof(Passenger.PassengerClass), nameof(Passenger.PassengerClass)))
                .Append(mlContext.Transforms.Concatenate("Features", nameof(Passenger.PassengerClass), nameof(Passenger.Gender),
                    nameof(Passenger.Age), nameof(Passenger.SiblingsOrSpouses),
                    nameof(Passenger.ParentsOrChildren), nameof(Passenger.Embarked)))
                //.Append(mlContext.BinaryClassification.Trainers.LogisticRegression(nameof(Passenger.Survived)))
                .Append(mlContext.BinaryClassification.Trainers.FastTree(nameof(Passenger.Survived)))
                .Fit(trainingData);

            EvaluateModel(mlContext, dataPipeline);

            Console.WriteLine();

            PredictTestData(mlContext, dataPipeline);

            Console.WriteLine();

            //SaveModel(mlContext, dataPipeline);
        }

    private static void EvaluateModel(MLContext mlContext, ITransformer dataPipeline)
    {
        var trainingData = mlContext.Data.ReadFromTextFile<Passenger>(_trainingData, hasHeader: true, separatorChar: ',');

        var statistics = mlContext.BinaryClassification.EvaluateNonCalibrated(dataPipeline.Transform(trainingData), nameof(OutputModel.Survived));

        Console.WriteLine("Training Performance: ");
        Console.WriteLine($"\tAccuracy: {statistics.Accuracy}");
        Console.WriteLine($"\tF1: {statistics.F1Score}");
    }

    private static void PredictTestData(MLContext mlContext, ITransformer dataPipeline)
    {
        var predictor = dataPipeline.CreatePredictionEngine<PredictedData, OutputModel>(mlContext);

        var evalData = mlContext.Data.ReadFromTextFile<PredictedData>(_testData, hasHeader: true, separatorChar: ',');

        var csvWriter = new StringBuilder();
        csvWriter.Append("PassengerId,Survived\n");

        Console.WriteLine($"PassengerId, Survived");

        foreach(var row in evalData.Preview(2000).RowView)
        {
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

            var prediction = predictor.Predict(inputModel);

            Console.WriteLine($"{inputModel.PassengerId}, {(prediction.Survived ? "1" : "0")}");
            var pId = inputModel.PassengerId;
            var _survived = prediction.Survived ? "1" : "0";
            var newLine = $"{pId},{_survived}\n";
            csvWriter.Append(newLine);
        }

        File.WriteAllText("Data/submission.csv", csvWriter.ToString());
    }

    private static void SaveModel(MLContext mlContext, ITransformer dataPipeline)
    {
        Console.WriteLine("Saving the model to Model.zip");
        using(var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
        {
            dataPipeline.SaveTo(mlContext, fs);
        }
    }
  }
}