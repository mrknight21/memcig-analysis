import json
import random


from pymongo import MongoClient
import os
from dotenv import load_dotenv
from mongo_users import add_new_user


# Load environment variables from the .env file
load_dotenv()

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")  # Default MongoDB URI
DB_NAME = os.getenv("DB_NAME", "flask_mongo_db")  # Default database name
INFO_STUDY_ID = os.getenv("INFO_STUDY_ID", "info")
MIX_STUDY_ID = os.getenv("MIX_STUDY_ID", "mix")
# MongoDB Client
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
users_collection = db['users']  # Users collection
tasks_collection = db['tasks']  # Tasks collection
tasks_assignments_collection = db['tasks_assignments']
studys_collection = db['studies']


max_annotation_per_task = 3

# Path to the test task JSON file
test_task_file = '../data/tasks/fora_tasks.json'


def load_tasks(file_path):
    """
    Load tasks from a JSON file.

    Args:
        file_path (str): Path to the file containing tasks in JSON format.

    Returns:
        list: A list of task objects (dictionary).
    """
    with open(file_path, 'r') as f:
        tasks = json.load(f)
    return tasks

def initiate_tasks(tasks, split="prod", filter=None):
    """
        Load tasks from a JSON file and upload them to MongoDB.
        """
    try:

        # Validate if tasks were loaded
        if not isinstance(tasks, list):
            raise ValueError("The JSON file must contain a list of task objects.")

        # Insert tasks into the MongoDB collection
        for task in tasks:
            task["split"] = split
        if filter:
            tasks = [task for task in tasks if not any([str(cov_id) in task["conversation_id"] for cov_id  in filter])]
        response = tasks_collection.insert_many(tasks)
        print(response)

    except Exception as e:
        print(f"Error occurred: {e}")

def initiate_test_user():
    add_new_user("test_user", db)

def initiate_study_metadata():
    study_info = {
        "study_id": INFO_STUDY_ID,
        "task_type": "info",
        "status": "active",
        "description": "In this study, we are interested in the audience's perceptions of the informativesness of the public conversations, such as debates and community discussion."
    }
    study_mix = {
        "study_id": MIX_STUDY_ID,
        "task_type": "mix",
        "status": "active",
        "description": "In this study, we are interested in the audience's perceptions of novelty, relevance, and scope of the public conversations, such as debates and community discussion."
    }
    studies = [study_info, study_mix]
    response = studys_collection.insert_many(studies)
    print("number of studies: ", response.inserted_ids.__len__(), "")


def initiate_task_assignment(split="dev", assigned_ids=None):
    tasks = tasks_collection.find({"corpus_id": "fora"})
    task_groups = {}
    for task in tasks:
        task_id =  task["task_id"]
        cov_id = task["conversation_id"]
        if assigned_ids:
            if not any([str(_id) in task["conversation_id"] for _id in assigned_ids]):
                continue
        if cov_id not in task_groups:
            task_groups[cov_id] = {
            "corpus_id": task["corpus_id"],
            "conversation_id": task["conversation_id"],
            "info_cur_annotators": [],
            "info_submitted_annotations": [],
            "mix_cur_annotators": [],
            "mix_submitted_annotations": [],
            "tasks": [],
            "info_complete": False,
            "mix_complete": False,
            "info_occupied": False,
            "mix_occupied": False,
            "split": split
        }
        task_groups[cov_id]["tasks"].append(task_id)

    for cov_id, task_group in task_groups.items():
        task_group["tasks"] = sorted(task_group["tasks"], key=lambda s: int(s.split('_')[-1]))
    response = tasks_assignments_collection.insert_many(list(task_groups.values()))
    print(response)




def setup_mongo_db():
    # Load tasks from the file
    dev_tasks = load_tasks(test_task_file)
    initiate_tasks(dev_tasks)
    # initiate_task_assignment()
    # initiate_study_metadata()
    # initiate_test_user()





if __name__ == '__main__':
    setup_mongo_db()
