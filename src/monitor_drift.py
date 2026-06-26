import os
import pandas as pd
import mlflow
from evidently.report import Report
from evidently.metric_preset import TextOverviewPreset

def run_monitoring():
    # 1. Setup MLflow Tracking
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("RAG_Production_Monitoring")
    
    # 2. Mocking Reference and Production datasets for Text Drift
    # In production, substitute these with your real validation data and live logs
    reference_data = pd.DataFrame({
        "text": [
            "What is the capital of France?",
            "How does photosynthesis work?",
            "Explain deep learning in simple terms.",
            "What is the distance to the moon?",
            "How do I install docker on windows?"
        ]
    })
    
    production_data = pd.DataFrame({
        "text": [
            "How can I deploy an app on Kubernetes?",
            "Tell me about Docker desktop errors.",
            "What is the best way to trace text drift?",
            "How do I configure an HPA autoscaler?",
            "Is Python or JavaScript better for AI?"
        ]
    })
    
    print("Evaluating data drift using Evidently...")
    
    # 3. Configure and Run Evidently Report
    text_report = Report(metrics=[
        TextOverviewPreset(column_name="text")
    ])
    
    text_report.run(reference_data=reference_data, current_data=production_data)
    
    # 4. Extract metrics to log into MLflow
    result = text_report.as_dict()
    
    # Extract structural summaries or drift scores depending on properties
    # For a simple demo pipeline, we log a run with metrics
    with mlflow.start_run():
        print("Logging metrics to MLflow tracking server...")
        
        # Example metrics extracted from report JSON structure
        # Evidently TextOverview metrics structure can be parsed or verified
        mlflow.log_param("data_source", "production_logs_v1")
        mlflow.log_metric("reference_sample_count", len(reference_data))
        mlflow.log_metric("production_sample_count", len(production_data))
        
        # Save HTML report as an artifact
        report_html_path = "evidently_report.html"
        text_report.save_html(report_html_path)
        mlflow.log_artifact(report_html_path)
        
        print("Monitoring run recorded successfully!")

if __name__ == "__main__":
    run_monitoring()