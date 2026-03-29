from azure.storage.blob import BlobServiceClient
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

class AzureStorageService:
    def __init__(self):
        self.connection_string = os.getenv("AZURE_CONNECTION_STRING")
        self.container_name = "documents"

        self.client = BlobServiceClient.from_connection_string(
            self.connection_string
        )
        self.container = self.client.get_container_client(self.container_name)

    def upload_file(self, file):
        import uuid

        unique_name = f"{uuid.uuid4()}_{file.filename}"

        blob_client = self.container.get_blob_client(unique_name)
        blob_client.upload_blob(file.file, overwrite=True)

        return unique_name
    
    def download_file(self,blob_name:str):
        blob_client = self.container.get_blob_client(blob_name)
        stream = blob_client.download_blob()
        return stream.readall()
    
    def delete_file(self, blob_name: str):
        try:
            blob_client = self.container.get_blob_client(blob_name)
            blob_client.delete_blob()
            print(f"Deleted from Azure: {blob_name}")
        except Exception as e:
            print(f"Azure delete failed: {e}")
            raise e