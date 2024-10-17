
"""

    The function `save_in_gcs` takes a file path, determines the file extension, uploads the file to
    Google Cloud Storage in a specific folder based on the file type, generates a signed URL for the
    uploaded file, and returns the URL.
    
    :param file_path: The `save_in_gcs` function you provided seems to be a Python function that saves a
    file to Google Cloud Storage (GCS) and generates a signed URL for accessing the file

    :return: The function `save_in_gcs` returns a signed URL that allows access to the uploaded file in
    Google Cloud Storage.

"""

from google.cloud import storage
from datetime import timedelta
from dotenv import load_dotenv
from datetime import datetime
import uuid
import os


load_dotenv()


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def save_in_gcs(file_path):

    file_extension = os.path.splitext(file_path)[1]
    print (file_extension)

    unique_filename = f"output_{timestamp}_{uuid.uuid4().hex[:6]}{file_extension}"
    print (unique_filename)

    if file_extension in [".png", ".jpg", ".jpeg", ".gif"]:
        folder_name = "Images"
    elif file_extension in [".mp4", ".mov", ".avi"]:
        folder_name = "Videos"
    elif file_extension in [".mp3", ".wav"]:
        folder_name = "Audios"
    
    print (folder_name)

    destination_blob_name = f"{folder_name}/{unique_filename}"
    print (destination_blob_name)

    storage_client = storage.Client()
    bucket_name = "personalized_video"
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(file_path)
    print(f"File {file_path} uploaded to {destination_blob_name}.")

    # os.remove(file_path)

    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(days=1),
        method="GET"
    )

    print(f"Generated signed URL: {url}")
    return url


# Example usage
# save_in_gcs("cropped.mp4")