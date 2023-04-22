from scrapinghub import ScrapinghubClient
import os
from dotenv import load_dotenv
import csv


file_path="/Users/samyuktha/recommender-system/data/"
# get all jobs for a spider

def get_items(project,spider_name):
    jobs = project.jobs.list(spider=spider_name)

    print(jobs)

    # get all keys from jobs

    keys= [job['key'] for job in jobs]


    # get latest job

    latest=keys[0]

  

    job = project.jobs.get(latest)

    items = job.items.iter()

    for item in items:

        decoded_dict = {key.decode('utf-8'): value.decode('utf-8') for key, value in item.items()}

        print(decoded_dict.values())

        with open(file_path+'data.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(list(decoded_dict.values())[:3])


def fetch_data():

    # Load environment variables from .env file

    load_dotenv()

    # Retrieve API key from environment variable
    api_key = os.environ.get('ZYTE_API')
    project_id = os.environ.get('PROJECT_ID')

    # Create a client

    client = ScrapinghubClient(api_key)


    # Get a project

    project = client.get_project(project_id)

    # get all spiders

    spiders = project.spiders.list()

    open(file_path+'data.csv', 'w').close()
    with open(file_path+'data.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['title','link','published_date'])
    for spider in spiders:
        get_items(project,spider['id'])




