"""
Data preparation pipeline step.
"""
import logging
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
# Import Azure ML Workspace lazily inside the function to avoid import-time
# failures in CI when azureml SDK or its dependencies are not available.
from src.components.logger import setup_logger, log_section, log_step
from src.components.data_loader import DataLoader, DataPreprocessor

logger = setup_logger(__name__, log_file="data_prep.log")


def run_data_prep(
    input_datastore_path: str,
    local_data_dir: str,
    output_dir: str,
    test_size: float = 0.2,
    random_state: int = 42,
    use_workspace: bool = True
):
    """
    Prepare data for training.
    
    Args:
        input_datastore_path: Path to input data in datastore
        local_data_dir: Local directory for downloads
        output_dir: Directory to save prepared data
        test_size: Fraction of data for test set
        random_state: Random seed
        use_workspace: Whether to use Azure ML workspace
    """
    log_section(logger, "DATA PREPARATION PIPELINE")
    
    try:
        # Initialize data loader
        workspace = None
        if use_workspace:
            from azureml.core import Workspace
            workspace = Workspace.from_config()
        loader = DataLoader(workspace=workspace)
        
        log_step(logger, "Loading data from Azure ML datastore")
        df = loader.load_from_datastore(
            datastore_path=input_datastore_path,
            local_download_dir=local_data_dir
        )
        
        # Validate data
        log_step(logger, "Validating data quality")
        DataPreprocessor.validate_data(df)
        
        # Split features and target
        log_step(logger, "Splitting features and target")
        X, y = DataPreprocessor.split_features_target(df, target_column="Class")
        
        # Train/test split
        log_step(logger, "Splitting into train/test sets")
        x_train, x_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        logger.info(f"Train set size: {len(x_train)}")
        logger.info(f"Test set size: {len(x_test)}")
        
        # Save prepared data
        log_step(logger, "Saving prepared data")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        import joblib
        joblib.dump(x_train, Path(output_dir) / "x_train.pkl")
        joblib.dump(x_test, Path(output_dir) / "x_test.pkl")
        joblib.dump(y_train, Path(output_dir) / "y_train.pkl")
        joblib.dump(y_test, Path(output_dir) / "y_test.pkl")
        
        logger.info(f"✅ Data preparation completed successfully")
        logger.info(f"Output saved to: {output_dir}")
        
        return {
            "status": "success",
            "train_size": len(x_train),
            "test_size": len(x_test),
            "features": len(X.columns),
            "output_dir": output_dir
        }
        
    except Exception as e:
        logger.error(f"❌ Data preparation failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-datastore-path", type=str, 
                       default="UI/2026-04-04_090211_UTC/creditcard_train.csv")
    parser.add_argument("--local-data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs/prepared_data")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--no-workspace",
        action="store_true",
        help="Do not attempt to load Azure ML Workspace (useful for CI runs)"
    )
    
    args = parser.parse_args()
    
    use_workspace_flag = not args.no_workspace

    result = run_data_prep(
        input_datastore_path=args.input_datastore_path,
        local_data_dir=args.local_data_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state
        , use_workspace=use_workspace_flag
    )
    
    import json
    with open(Path(args.output_dir) / "data_prep_result.json", "w") as f:
        json.dump(result, f, indent=2)
