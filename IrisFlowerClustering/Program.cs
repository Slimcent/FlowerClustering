using IrisFlowerClustering;
using Microsoft.ML;

string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "iris.csv");
string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "IrisClusteringModel.zip");

//string dataFolderPath = Path.Combine(Directory.GetCurrentDirectory(), "Data");
string dataFolderPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Data");
string csvFilePath = Path.Combine(dataFolderPath, "iris.csv");

var mlContext = new MLContext(seed: 0);

IDataView dataView = mlContext.Data.LoadFromTextFile<IrisData>(csvFilePath, hasHeader: false, separatorChar: ',');

// Create a learning pipeline
string featuresColumnName = "Features";
var pipeline = mlContext.Transforms
    .Concatenate(featuresColumnName, "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
    .Append(mlContext.Clustering.Trainers.KMeans(featuresColumnName, numberOfClusters: 3));

// Train the model
var model = pipeline.Fit(dataView);

//Save the model
using (var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
{
    mlContext.Model.Save(model, dataView.Schema, fileStream);
}

// Use model for prediction
var predictor = mlContext.Model.CreatePredictionEngine<IrisData, ClusterPrediction>(model);

var prediction = predictor.Predict(TestIrisData.Setosa);
Console.WriteLine($"Cluster: {prediction.PredictedClusterId}");
Console.WriteLine($"Distances: {string.Join(" ", prediction.Distances ?? Array.Empty<float>())}");

