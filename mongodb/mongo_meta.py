
def get_study_meta(study_id, db):
    return db.studies.find_one({'study_id': study_id})