import boto3
import pandas as pd
from io import StringIO

from src.logger.logging_file import logger


class s3_operations:
    def __init__(
        self,
        bucket_name,
        aws_access_key=None,
        aws_secret_key=None,
        region_name=None,
    ):
        """
        Initialize S3 client with explicit credentials or default AWS credential chain.
        """
        self.bucket_name = bucket_name

        client_kwargs = {"service_name": "s3"}
        if region_name:
            client_kwargs["region_name"] = region_name

        # If credentials are omitted, boto3 uses default chain:
        # env vars, shared config/profile, or IAM role.
        if aws_access_key and aws_secret_key:
            client_kwargs["aws_access_key_id"] = aws_access_key
            client_kwargs["aws_secret_access_key"] = aws_secret_key

        self.s3_client = boto3.client(**client_kwargs)
        logger.info("Data Ingestion from S3 bucket initialized")

    def fetch_file_from_s3(self, file_key):
        """
        Fetches a CSV file from the S3 bucket and returns it as a Pandas DataFrame.
        :param file_key: S3 file path (e.g., 'data/data.csv')
        :return: Pandas DataFrame
        """
        try:
            logger.info(
                "Fetching file '%s' from S3 bucket '%s'...",
                file_key,
                self.bucket_name,
            )
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_key)
            df = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))
            logger.info(
                "Successfully fetched and loaded '%s' from S3 that has %s records.",
                file_key,
                len(df),
            )
            return df
        except Exception as e:
            logger.exception("Failed to fetch '%s' from S3: %s", file_key, e)
            return None
