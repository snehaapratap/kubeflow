from typing import Dict, List
from kfp import dsl
from kfp import compiler
import kfp
from kfp.dsl import Input, Output, Dataset, Model, component


@dsl.component(base_image="python:3.9")
def load_data(output_csv: Output[Dataset]):
    import subprocess
    subprocess.run(["pip", "install", "pandas", "scikit-learn"], check=True)

    from sklearn.datasets import load_iris
    import pandas as pd
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target


    df.to_csv(output_csv.path, index=False)


@dsl.component(base_image="python:3.9")
def preprocess_data(input_csv: Input[Dataset], output_train: Output[Dataset], output_test: Output[Dataset], 
                    output_ytrain: Output[Dataset], output_ytest: Output[Dataset]):
    import subprocess
    subprocess.run(["pip", "install", "pandas", "scikit-learn"], check=True)

    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    # Load dataset
    df = pd.read_csv(input_csv.path)


    print("Initial dataset shape:", df.shape)
    print("Missing values before preprocessing:\n", df.isnull().sum())


    if df.isnull().values.any():
        print("Missing values detected. Handling them...")
        df = df.dropna()  
    

    assert not df['target'].isnull().any(), "Target column contains NaN values after handling missing values."

    features = df.drop(columns=['target'])
    target = df['target']


    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)


    X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, random_state=42)


    print("Shapes after train-test split:")
    print("X_train:", X_train.shape, "X_test:", X_test.shape) 
    print("y_train:", y_train.shape, "y_test:", y_test.shape)
    print("Missing values in y_train:", y_train.isnull().sum())


    assert not y_train.isnull().any(), "y_train contains NaN values."
    assert not y_test.isnull().any(), "y_test contains NaN values."


    X_train_df = pd.DataFrame(X_train, columns=features.columns)
    print("X_train_df:", X_train_df) 

    y_train_df = pd.DataFrame(y_train) 
    print("y_train_df: ", y_train_df)  

    X_test_df = pd.DataFrame(X_test, columns=features.columns)
    print("X_test_df:", X_test_df) 

    y_test_df = pd.DataFrame(y_test) 
    print("y_test_df: ", y_test_df) 


    X_train_df.to_csv(output_train.path, index=False)  
    X_test_df.to_csv(output_test.path, index=False)

    y_train_df.to_csv(output_ytrain.path, index=False)  
    y_test_df.to_csv(output_ytest.path, index=False) 


@dsl.component(base_image="python:3.9")
def train_model(train_data: Input[Dataset], ytrain_data: Input[Dataset], model_output: Output[Model]):

    import subprocess
    subprocess.run(["pip", "install", "pandas", "scikit-learn", "joblib"], check=True)

    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from joblib import dump


    train_df = pd.read_csv(train_data.path)
    print("Shape of train_df:", train_df.shape)
    print("train_df:", train_df)
    X_train = train_df 

    y_train = pd.read_csv(ytrain_data.path)
    print("Shape of ytrain_df:", y_train.shape)
    print("y_train_df:", y_train)


    print("Shapes of X_train and y_train: ")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape) 
    print("Missing values in X_train:", X_train.isnull().sum())
    print("Missing values in y_train:", y_train.isnull().sum()) 


    assert not X_train.isnull().values.any(), "X_train contains NaN values."
    assert not y_train.isnull().values.any(), "y_train contains NaN values." 


    model = LogisticRegression()
    model.fit(X_train, y_train)


    dump(model, model_output.path)


@dsl.component(base_image="python:3.9")
def evaluate_model(test_data: Input[Dataset], ytest_data: Input[Dataset], model: Input[Model], metrics_output: Output[Dataset]):
    import subprocess
    subprocess.run(["pip", "install", "pandas", "scikit-learn", "matplotlib", "joblib"], check=True)

    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    from joblib import load


    X_test = pd.read_csv(test_data.path)

    y_test = pd.read_csv(ytest_data.path)  


    model = load(model.path)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)


    metrics_path = metrics_output.path
    with open(metrics_path, 'w') as f:
        f.write(str(report))


    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(metrics_path.replace('.txt', '.png'))


@dsl.pipeline(name="ml-pipeline")
def ml_pipeline():
    load_op = load_data()
    preprocess_op = preprocess_data(input_csv=load_op.outputs["output_csv"])
    train_op = train_model(train_data=preprocess_op.outputs["output_train"], ytrain_data=preprocess_op.outputs["output_ytrain"])
    evaluate_op = evaluate_model(test_data=preprocess_op.outputs["output_test"], ytest_data=preprocess_op.outputs["output_ytest"], model=train_op.outputs["model_output"]) 

if __name__ == "__main__":
    compiler.Compiler().compile(pipeline_func=ml_pipeline, package_path="kubeflow_pipeline.yaml")
