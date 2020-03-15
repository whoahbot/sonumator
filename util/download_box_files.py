import os
import sys
from concurrent.futures import ThreadPoolExecutor
from subprocess import call, check_output

from boxsdk import JWTAuth, Client

def download_file(client, file_id, output_file):
    with open(output_file, "wb") as f:
        client.file(file_id).download_to(f)
        
    return_code = call(["gsutil", "cp", output_file, f"gs://sonumator/recordings/{output_file}"])
    os.remove(output_file)

if __name__ == "__main__":
    auth = JWTAuth.from_settings_file('settings.json')
    client = Client(auth)

    folder_2017_2018 = client.folder(folder_id='106561420338').get()
    items = folder_2017_2018.get_items()

    output_directory = "2016-2017"
    with ThreadPoolExecutor(max_workers=10) as executor:
        res = executor.map(lambda item: download_file(client, item.id, f"{output_directory}/{item.name}"), items)
