using Microsoft.ML.Data;

namespace TitanicML
{
    public class OutputModel
    {
        [ColumnName("PredictedLabel")]
        public bool Survived;

        [ColumnName("Probability")]
        public float Probability;
    }
}