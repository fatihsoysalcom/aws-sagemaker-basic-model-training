import sagemaker
import boto3
from sagemaker.amazon.amazon_estimator import get_image_uri

# AWS SDK client for SageMaker
sagemaker_client = boto3.client('sagemaker')

# Get the SageMaker execution role ARN. This role needs permissions to access S3 and SageMaker.
# Replace with your actual SageMaker execution role ARN.
# You can find this in the AWS IAM console.
role = sagemaker.get_execution_role()

# Specify the AWS region
region = boto3.Session().region_name

# Get the built-in XGBoost algorithm image URI for the specified region
container = get_image_uri(region, 'xgboost', '1.7-1')

# Define the SageMaker session
session = sagemaker.Session()

# Define the training job name
job_name = 'xgboost-basic-training-job'

# Define S3 input data location (replace with your actual data path)
# For this example, we'll assume you have a file named 'train.csv' in an S3 bucket.
# You would typically upload your training data to S3 first.
s3_input_train = sagemaker.inputs.TrainingInput(
    s3_data='s3://your-sagemaker-bucket/path/to/your/train.csv',
    content_type='csv',
    distribution='FullyReplicated',
    s3_data_type='S3Prefix'
)

# Define the SageMaker Estimator for XGBoost
xgboost_estimator = sagemaker.estimator.Estimator(
    image_uri=container,
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    output_path=f's3://your-sagemaker-bucket/sagemaker-output/{job_name}',
    sagemaker_session=session,
    hyperparameters={
        'max_depth': 5,
        'eta': 0.2,
        'gamma': 4,
        'min_child_weight': 6,
        'subsample': 0.8,
        'verbosity': 1,
        'objective': 'reg:squarederror',
        'num_round': 100
    }
)

# Start the training job
print(f"Starting training job: {job_name}")
xgboost_estimator.fit({'train': s3_input_train})

print(f"Training job {job_name} completed.")
print(f"Model artifacts are stored in: {xgboost_estimator.model_data}")
