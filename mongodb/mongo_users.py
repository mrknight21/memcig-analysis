
def add_new_user(prolific_id, db):
    users_collection = db['users']
    user = {
        "prolific_id": prolific_id,
        "completed_prescreen_info": False,
        "completed_prescreen_mix": False,
        "consent_signed_timestamp": None,
        "cur_study": None,
        "cur_session": None,
        "cur_task": None,
        "cur_conversation_id": None,
        "cur_task_assigned_time": None,
        "progress": 0,
        "completed_tasks": [],
        "completed_task_groups": [],
    }
    users_collection.insert_one(user)
    return user

def get_user_by_id(prolific_id, db):
    users_collection = db['users']
    user = users_collection.find_one({'prolific_id':prolific_id})
    return user

def update_user_cur_study(prolific_id, study_id, db):
    users_collection = db['users']
    resp = users_collection.update_one(
        {'prolific_id': prolific_id},
        {'$set': {'cur_study': study_id}}
    )
    print(f"Updated user {prolific_id} with cur study {study_id}")
    return resp.modified_count

def update_user_consent_timestamp(username, timestamp, db):
    users_collection = db['users']
    resp = users_collection.update_one(
        {'prolific_id': username},
        {'$set': {'consent_signed_timestamp': timestamp}}
    )
    print(f"Updated user {username} with consent timestamp {timestamp}")
    return resp.modified_count

def update_user_cur_session(username, session_id, db):
    users_collection = db['users']
    resp = users_collection.update_one(
        {'prolific_id': username},
        {'$set': {'cur_session': session_id}}
    )
    print(f"Updated user {username} with cur session {session_id}")
    return resp.modified_count

def update_user_cur_tasks(username, task_id, db):
    users_collection = db['users']
    resp = users_collection.update_one(
        {'prolific_id': username},
        {'$set': {'cur_tasks': task_id}}
    )
    print(f"Updated user {username} with cur tasks {task_id}")
    return resp.modified_count

def update_user_completed_tasks(username, task_id, db):
    users_collection = db['users']
    resp = users_collection.update_one(
        {'prolific_id': username},
        {'$push': {'completed_tasks': task_id}}
    )
    print(f"Updated user {username} with completed tasks {task_id}")
    return resp.modified_count

def update_user_completed_prescreen(username, db, task_type="info"):
    users_collection = db['users']
    resp = users_collection.update_one(
        {'prolific_id': username},
        {'$set': {f'completed_prescreen_{task_type}': True}}
    )
    print(f"Updated user {username} with completed prescreen")
    return resp.modified_count

