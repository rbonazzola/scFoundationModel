import mlflow

def mlflow_log_params(params):
    for key, value in params.items():
        mlflow.log_param(key, value)

def mlflow_log_metrics(metrics):
    for key, value in metrics.items():
        mlflow.log_metric(key, value)

def mlflow_log_artifacts(artifacts):
    for artifact in artifacts:
        mlflow.log_artifact(artifact)

def mlflow_log_model(model, model_name):
    mlflow.pytorch.log_model(model, model_name)

def mlflow_set_experiment(experiment_name):
    mlflow.set_experiment(experiment_name)

def mlflow_start_run(run_name):
    mlflow.start_run(run_name=run_name)