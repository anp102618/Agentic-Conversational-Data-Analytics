import json
import pandas as pd
from pathlib import Path
from groq import Groq
import src
from src.config import config, BASE_DIR


def analyze_eda_columns_with_groq(df: pd.DataFrame, api_key: str, model: str = config.GROQ_MODEL) -> dict:

  """Samples a pandas DataFrame and uses Groq to classify columns

  for numeric, categorical, datetime, and EDA types (uni, bi, multivariate).

  Returns a dictionary of lists.
  """
  # Initialize the Groq client
  client = Groq(api_key=api_key)

  # Take a small sample of the dataframe to save tokens (e.g., 5 rows)
  sample_df = df.sample(n=min(5, len(df)), random_state=42)
  data_sample = sample_df.to_csv(index=False)

  # Define the prompt engineering for structured JSON output
  prompt = f"""
    Analyze the following CSV sample of a dataset and its column headers. 
    Classify the columns based on their data types and suitability for Exploratory Data Analysis (EDA).
    
    Data Sample:
    {data_sample}
    
    You must return a valid JSON object ONLY, with no extra text or markdown code formatting blocks outside of the JSON if possible, containing these exact keys as lists of column names:
    - "univariate_candidates": columns best suited for single-variable distribution plots (histograms, boxplots).
    - "bivariate_candidates": pairs or individual columns heavily suited for relationship analysis (scatter plots, correlation).
    - "multivariate_candidates": columns that can be used together in complex multi-variable analysis (e.g., pairplots, heatmaps, cluster analysis).
    - provide the candidates for the above three types of analysis in the form of a dictionary where numeric , categorical and datetime columns are also provided as lists of column names.
    """
  # Call the Groq Chat Completion API
  response = client.chat.completions.create(
      model=model,
      messages=[
          {
              "role": "system",
              "content": (
                  "You are an expert data scientist. Return strictly valid"
                  " JSON containing a dictionary of lists."
              ),
          },
          {"role": "user", "content": prompt},
      ],
      response_format={"type": "json_object"},
      temperature=0.1,
  )

  # Parse and return the JSON response as a Python dictionary
  result_content = response.choices[0].message.content
  return json.loads(result_content)


def classify_columns(file_path: Path= BASE_DIR / "src" / "Data" / "cleaned_ecommerce_dataset.csv" ):
  
    output_json_path = BASE_DIR / "src" / "Data" / "classified_columns.json"
    df = pd.read_csv(file_path)
    eda_dict = analyze_eda_columns_with_groq(df, api_key=config.GROQ_API_KEY)
    with open(output_json_path, "w") as outfile:
        json.dump(eda_dict, outfile, indent=4)
    return eda_dict