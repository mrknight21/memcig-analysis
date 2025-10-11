import random
from datetime import datetime, timezone

TASK_TARGET = 3

def get_participant_full_count(task_assignment, task_type):
    return len(task_assignment[f"{task_type}_cur_annotators"]) + len(task_assignment[f"{task_type}_submitted_annotations"])



def decide_task_type(task_assignment):
    info_count = len(task_assignment["info_cur_annotators"]) + len(task_assignment["info_submitted_annotations"])
    mix_count = len(task_assignment["mix_cur_annotators"]) + len(task_assignment["mix_submitted_annotations"])
    if info_count <= mix_count:
        return "info"
    else:
        return "mix"

def get_task_assignment(conv_id, db):
    tasks_assignments_collection = db['tasks_assignments']
    task_assignment = tasks_assignments_collection.find_one({'conversation_id': conv_id})
    return task_assignment


def get_task_by_id(task_id, db):
    tasks_collection = db['tasks']
    task = tasks_collection.find_one({'task_id': task_id})
    return task


def get_available_task_group(user, task_type, db):
    tasks_assignments_collection = db['tasks_assignments']
    task_statuses = list(tasks_assignments_collection.find({f"{task_type}_complete": False, f"{task_type}_occupied": False}))

    available_task = [status for status in task_statuses if len(status[f"{task_type}_cur_annotators"]) +
                      len(status[f"{task_type}_submitted_annotations"]) < TASK_TARGET]

    if len(available_task) == 0:
        return None, None

    # This algorithm prioritise task that are almost completed, since we need at least N annotator per task.
    for availability_tier in range(1, TASK_TARGET+1):
        tasks_in_tier = []

        for status in available_task:
            total_participants = get_participant_full_count(status, task_type)
            if TASK_TARGET - total_participants == availability_tier:
                tasks_in_tier.append(status)

        if len(tasks_in_tier) > 0:
            random.shuffle(tasks_in_tier)
            for candidate in tasks_in_tier:
                completed_tasks = [t for t in user["completed_tasks"] if task_type in t]
                incomplete_tasks = [t for t in candidate["tasks"] if t + f"_{task_type}" not in completed_tasks]
                if len(incomplete_tasks) == len(candidate["tasks"]):
                    return candidate["conversation_id"], candidate
    return None, None


def assign_task_to_user(user, task_type, db, task_group_id=None, task_id=None):

    completion_status = False
    progress = 0
    users_collection = db['users']
    tasks_assignments_collection = db['tasks_assignments']

    if not task_group_id:
        task_group_id, task_group = get_available_task_group(user, task_type, db)
    else:
        task_group = tasks_assignments_collection.find_one({'conversation_id': task_group_id})
        
    if not task_group:
        return task_group_id, task_id, user, completion_status

    if not task_id:
        completed_task = [t for t in user["completed_tasks"] if task_type in t]
        available_task = [t + f"_{task_type}" for t in task_group["tasks"] if t + f"_{task_type}" not in  completed_task]
        if len(available_task) != 0:
            task_id = available_task[0]
            progress = (len(task_group["tasks"]) - len(available_task)) / len(task_group["tasks"])
        else:
            return task_group_id, task_id, completion_status

    user["cur_task"] = task_id
    user["cur_conversation_id"] = task_group_id
    user["progress"] = progress

    user_resp = users_collection.update_one({"prolific_id": user["prolific_id"]},
                                {"$set": user})

    task_group[f"{task_type}_cur_annotators"].append(user["prolific_id"])
    task_group = update_task_assignment_status(task_group, task_type)

    task_assignment_resp = tasks_assignments_collection.update_one({'conversation_id': task_group_id},
                                            {"$set": task_group})
    if task_assignment_resp.modified_count == 1 and user_resp.modified_count == 1:
        completion_status = True
    return task_group_id, task_id, user, completion_status


def submit_annotation(annotation, task_id, task_type, user, db):
    completion = False
    tasks_assignments_collection = db['tasks_assignments']
    users_collection = db['users']
    annotation_collection = db['annotations']

    # the format of submission id
    submission_id = "_".join([task_id, user["cur_session"], user["prolific_id"]])
    completion_id = None

    # update annotation
    annotation["submission_id"] = submission_id
    annotation["annotator"] = user["prolific_id"]
    annotation["type"] = task_type
    annotation["task_id"] = task_id
    annotation["session_id"] = user["cur_session"]
    annotation["submission_timestamp"] = datetime.now()
    annotation["conversation_id"] = user["cur_conversation_id"]
    annotation["study_id"] = user["cur_study"]
    annotation["status"] = "pending"
    annotation_db_resp = annotation_collection.insert_one(annotation)

    # get task assignment
    task_group = get_task_assignment(user["cur_conversation_id"], db)

    # update user
    user["completed_tasks"].append(task_id)
    user["cur_task"] = None

    completed_task = [t for t in user["completed_tasks"] if task_type in t]
    available_task = [t + f"_{task_type}" for t in task_group["tasks"] if t + f"_{task_type}" not in completed_task]
    progress = (len(task_group["tasks"]) - len(available_task)) / len(task_group["tasks"])
    user["progress"] = progress

    if progress == 1:
        completion = True
        completion_id = "_".join([user["cur_conversation_id"], user["cur_session"], user["prolific_id"]])
        task_group[f"{task_type}_cur_annotators"].remove(user["prolific_id"])
        task_group[f"{task_type}_submitted_annotations"].append(completion_id)

    #Assign new task
    if len(available_task) > 0:
        task_id = available_task[0]
        user["cur_task"] = task_id
        user["cur_task_assigned_time"] = datetime.now(timezone.utc) \
    .isoformat(timespec="milliseconds") \
    .replace("+00:00", "Z")

    task_group = update_task_assignment_status(task_group, task_type)

    users_collection.update_one({"prolific_id": user["prolific_id"]}, {"$set": user})
    tasks_assignments_collection.update_one({"conversation_id": task_group["conversation_id"]}, {"$set": task_group})


    return user, completion, completion_id, submission_id,


def update_task_assignment_status(task_group, task_type):
    if get_participant_full_count(task_group, task_type) == TASK_TARGET:
        task_group[f"{task_type}_occupied"] = True
    else:
        task_group[f"{task_type}_occupied"] = False

    if len(task_group[f"{task_type}_submitted_annotations"]) == TASK_TARGET:
        task_group[f"{task_type}_complete"] = True
    else:
        task_group[f"{task_type}_complete"] = False
    return task_group

def remove_user_from_task(user, conv_id, task_type, db):
    tasks_assignments_collection = db['tasks_assignments']
    users_collection = db['users']
    task_assignment = tasks_assignments_collection.find_one({"conversation_id": conv_id})
    task_assignment[f"{task_type}_cur_annotators"].remove(user["prolific_id"])

    task_assignment = update_task_assignment_status(task_assignment, task_type)

    task_assignment_resp = tasks_assignments_collection.update_one({"conversation_id": conv_id}, {"$set": task_assignment})
    user["cur_task"] = None
    user["cur_conversation_id"] = None
    user["progress"] = 0
    user["cur_task_assigned_time"] = None
    user_resp = users_collection.update_one({"prolific_id": user["prolific_id"]}, {"$set": user})
    if task_assignment_resp.modified_count == 1 and user_resp.modified_count == 1:
        return True
    else:
        return False

def insert_prescreen_input(input, db):
    prescreen_collection = db['prescreen']
    presreen_resp = prescreen_collection.insert_one(input)
    return presreen_resp.inserted_id


