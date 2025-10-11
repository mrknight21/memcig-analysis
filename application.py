
import datetime as dt
from flask import Flask, render_template, request, make_response, redirect, url_for, jsonify

from pymongo import MongoClient
import os
import urllib.parse

# Load environment variables if you're using a .env file (Optional)
from dotenv import load_dotenv

from mongodb.mongo_users import *
from mongodb.mongo_tasks import *
from mongodb.mongo_meta import *
from util import *

load_dotenv()



application = Flask(__name__)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "flask_mongo_db")
COMPLETION_CODE = os.getenv("COMPLETION_CODE", "1234567890")
PORT = os.getenv("PORT", "8000")
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
MODE = os.getenv("MODE", "DEBUG")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

prescreen_task_file = "static/prescreen_sample.json"
prescreen_task = load_json(prescreen_task_file)
corpus_meta = load_json("static/corpus_meta.json")
validation_threshold = 0.8
application.logger.setLevel(LOG_LEVEL)

def get_nav_info(user, study):
    nav_info = {
            "current_user": None,
            "study_type": None,
            "total_submissions": 0,
            "task_id": None,
        }
    if user:
        nav_info["current_user"] = user["prolific_id"]
        nav_info["total_submissions"] = len(user["completed_tasks"])
        nav_info["task_id"] = user["cur_task"]
        nav_info["conversation_id"] = user.get("cur_conversation_id", None)
        nav_info["progress"] = user.get("progress", 0) * 100
    if study:
        nav_info["study_type"] = study["task_type"]
    return nav_info


# Retreive information for user, session, study.
def get_state_info(request, db):
    user = None
    prolific_id = request.args.get('PROLIFIC_PID', default=None)
    study_id = request.args.get('STUDY_ID', default=None)
    session_id = request.args.get('SESSION_ID')

    if not prolific_id:
        prolific_id = request.cookies.get('prolific_id')
    if not study_id:
        study_id = request.cookies.get('study_id')
    if not session_id:
        session_id = request.cookies.get('session_id')

    if prolific_id:
        user = get_user_by_id(prolific_id, db)

    if user and not session_id:
        session_id = user.get("cur_session")

    if user and not study_id:
        study_id = user.get("cur_study")

    if not session_id:
        session_id = generate_session_token()
        if user:
            update_user_cur_session(prolific_id, session_id, db)
            user["cur_session"] = session_id
    elif user and session_id != user.get("cur_session"):
        update_user_cur_session(prolific_id, session_id, db)
        user["cur_session"] = session_id

    if study_id:
        study = get_study_meta(study_id, db)
        if user and study_id != user.get("cur_study"):
            update_user_cur_study(prolific_id, study_id, db)
            user["cur_study"] = study_id
    else:
        study = None

    return prolific_id, study_id, session_id, user, study


def evaluate_prescreen(user_input, task_utterances):
    annotation_utterances = user_input["target_utterances"]
    ground_truth_utterances = {str(utt["utterance_id"]): utt["answers"] for utt in task_utterances}
    ground_truth_hints = {str(utt["utterance_id"]): utt["hints"] for utt in task_utterances}
    scores = []
    error_messages = []
    report = {"score": 0, "message": "Something went wrong, perhaps invalid input.", "valid": False}
    if len(annotation_utterances) != len(task_utterances):
        return report
    else:
        for utt in annotation_utterances:
            utt_id = str(utt["utterance_id"])
            user_labels = utt["labels"]
            hints = ground_truth_hints[utt_id]
            if utt_id not in ground_truth_utterances:
                return report
            truth_labels = ground_truth_utterances[utt_id]
            for aspect, value in user_labels.items():
                hint = hints[aspect]
                distance = abs(truth_labels[aspect] - value)
                if distance == 1:
                    error_messages.append(f"<strong>Id {utt_id}</strong> slightly wrong for <strong>{aspect}</strong>: {hint}")
                if distance > 1:
                    error_messages.append(f"<strong>Id {utt_id}</strong> wrong for <strong>{aspect}</strong>: {hint}")
                distance = distance / 4
                score = 1 - distance
                scores.append(score)

    performance_score = round(sum(scores)/ len(scores), 2)
    message = "<br><br>".join(error_messages)
    message += f"<br>Total score: <strong>{performance_score}</strong>"
    report["score"] = performance_score
    report["message"] = message
    return report

def validate_annotation(user_input, task_utterances, task_type):
    annotation_utterances = user_input["target_utterances"]
    ground_truth_utterances = [str(utt["utterance_id"]) for utt in task_utterances if utt["skipped"] == False]
    if len(annotation_utterances) != len(task_utterances):
        return False
    else:
        for utt in annotation_utterances:
            utt_id = utt["utterance_id"]
            if "skipped" in utt and utt["skipped"] == True:
                continue
            user_labels = utt["labels"]
            if str(utt_id) not in ground_truth_utterances:
                return False
            if task_type == "mix":
                if any(aspect not in user_labels for aspect in ["novelty", "relevance", "scope"]):
                    return False
            else:
                if "informativeness" not in user_labels:
                    return False
    return True


@application.route('/')
def index():

    prolific_id, study_id, session_id, user, study = get_state_info(request, db)

    if not study:
        return render_template('welcome.html')
    elif study["status"] != "active":
        return redirect(
            url_for('error') + '?message=' + urllib.parse.quote(f"Study(ID:{study_id}) is currently not active!"))

    if not user:
        if not prolific_id:
            resp = make_response(redirect("register"))
            resp.set_cookie('study_id', study_id)
            if session_id:
                resp.set_cookie('session_id', session_id)
            return resp
        else:
            application.logger.info(f"User {prolific_id} not found in the database. Creating a new user.")
            user = add_new_user(prolific_id, db)
            if user:
                update_user_cur_study(prolific_id, study_id, db)
                application.logger.info(f"User {prolific_id} registered successfully.")

    if user.get("consent_signed_timestamp", None) is None:
        resp = make_response(redirect('pls_page'))
        resp = set_ids_cookies(resp, prolific_id, study_id, session_id)
        return resp

    resp = make_response(redirect("tutorial"))
    resp = set_ids_cookies(resp, prolific_id, study_id, session_id)
    return resp

@application.route("/debug/echo")
def echo():
    from flask import request
    application.logger.info("URL=%s  QS=%s  ARGS=%s",
                    request.url,
                    request.query_string,
                    request.args.to_dict(flat=False))
    return {
        "url": request.url,
        "args": request.args
    }


def set_ids_cookies(resp, prolific_id, study_id, session_id):
    resp.set_cookie('prolific_id', prolific_id)
    resp.set_cookie('study_id', study_id)
    resp.set_cookie('session_id', session_id)
    return resp

@application.route('/error')
def error():
    message = request.args.get('message', "An unexpected error occurred.")
    return render_template('error.html', error_message=message)

@application.route('/register', methods=['GET', 'POST'])
def register():
    prolific_id, study_id, session_id, user, study = get_state_info(request, db)
    if not study:
        return redirect(url_for('error') + '?message=' + urllib.parse.quote("Study not found or Study Id not provided!"))
    elif study["status"] != "active":
        return redirect(
            url_for('error') + '?message=' + urllib.parse.quote(f"Study(ID:{study_id}) is currently not active!"))

    if request.method == 'POST' and (request.form.get('existing_id') or request.form.get('registration_id')):

        # processing login
        if request.form.get('existing_id'):
            username = request.form.get('existing_id')
            existing_user = get_user_by_id(username, db)
            if existing_user:
                prolific_id = existing_user.get("prolific_id")
                if existing_user.get("consent_signed_timestamp", None) is None:
                    resp = make_response(redirect(url_for('pls_page')))
                else:
                    resp = make_response(redirect(url_for('tutorial')))
                resp = set_ids_cookies(resp, prolific_id, study_id, session_id)
                return resp
            else:
                return render_template('registration.html', message="loging failed with non-existing user!")

        # processing registration
        if request.form.get('registration_id'):
            username = request.form.get('registration_id')
            existing_user = get_user_by_id(username, db)
            if not existing_user:
                user = add_new_user(username, db)
                update_user_cur_study(username, study_id, db)
                if user:
                    application.logger.info(f"User {username} registered successfully.")
                    resp = make_response(redirect(url_for('pls_page')))
                    prolific_id = username
                    resp = set_ids_cookies(resp, prolific_id, study_id, session_id)
                    return resp
            else:
                # Add code to save the username after further checks
                return render_template('registration.html', message="registration failed with existing user!") # Redirect or render a success page as needed
    return render_template('registration.html')

@application.route('/pls_page', methods=['GET', 'POST'])
def pls_page():
    prolific_id, study_id, session_id, user, study = get_state_info(request, db)
    if not study:
        return redirect(url_for('error') + '?message=' + urllib.parse.quote("Study not found or Study Id not provided!"))
    elif study["status"] != "active":
        return redirect(
            url_for('error') + '?message=' + urllib.parse.quote(f"Study(ID:{study_id}) is currently not active!"))

    if request.method == 'POST':
        # 1. Retrieve the timestamp from the form
        timestamp = request.form.get('consent_timestamp', None)

        # 2. Identify the current user (e.g., from session)
        username = request.cookies.get('prolific_id')
        if not username:
            return redirect(url_for('register'))

        # 3. Update the user record in your database
        # Example: you might have a function update_user_consent_timestamp(username, timestamp)
        update_user_consent_timestamp(username, timestamp, db)

        # 4. Redirect to the next page, for example the tutorial
        return redirect(url_for('tutorial'))
    return render_template('pls_page.html')

@application.route('/tutorial', methods=['GET', 'POST'])
def tutorial():
    prolific_id, study_id, session_id, user, study = get_state_info(request, db)
    if not study:
        return redirect(
            url_for('error') + '?message=' + urllib.parse.quote("Study not found or Study Id not provided!"))
    elif study["status"] != "active":
        return redirect(
            url_for('error') + '?message=' + urllib.parse.quote(f"Study(ID:{study_id}) is currently not active!"))
    if not user:
        # User not registered => go to registration
        return redirect(url_for('register'))
    if not user["consent_signed_timestamp"]:
        return make_response(redirect(url_for('pls_page')))

    task_type = study["task_type"]

    if request.method == 'POST':

        prescreen_completed = user.get('completed_tasks', False)
        if not prescreen_completed:
            return redirect(url_for('prescreen'))
        else:
            return redirect(url_for('annotation'))

    # GET: render tutorial
    nav_info = get_nav_info(user, study)
    if task_type == "info":
        return render_template('tutorial_info.html', nav_info= nav_info, codebook=False)
    else:
        return render_template('tutorial_mix.html', nav_info= nav_info, codebook=False)


@application.route('/prescreen', methods=['GET', 'POST'])
def prescreen():
    prolific_id, study_id, session_id, user, study = get_state_info(request, db)
    if not study:
        return redirect(
            url_for('error') + '?message=' + urllib.parse.quote("Study not found or Study Id not provided!"))
    elif study["status"] != "active":
        return redirect(
            url_for('error') + '?message=' + urllib.parse.quote(f"Study(ID:{study_id}) is currently not active!"))

    if not user:
        resp =  make_response(redirect(url_for('register')))
        resp = set_ids_cookies(resp, prolific_id, study_id, session_id)
        return resp
    if not user["consent_signed_timestamp"]:
        resp = make_response(redirect(url_for('pls_page')))
        resp = set_ids_cookies(resp, prolific_id, study_id, session_id)
        return resp

    task_type = study["task_type"]
    corpus_intro = corpus_meta["insq"]["intro"]

    completed_prescreen = user.get(f'completed_prescreen_{task_type}', False)
    if request.method == 'POST':
        user_annotation = request.get_json()
        user_annotation["date_time"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user_annotation["user_id"] = prolific_id

        report = evaluate_prescreen(user_annotation, prescreen_task["target_utterances"])
        if report['score'] > 0.3:
            insert_prescreen_input(user_annotation, db)
        if report['score'] >= validation_threshold:
            update_user_completed_prescreen(prolific_id, db, task_type)
            application.logger.info(
                f"Prescreen for task {prescreen_task['task_id']} completed successfully for user {prolific_id}."
            )
            return jsonify({'success': True, 'message': report['message']})
        else:
            return jsonify({'success': False, 'message': report['message']})

    if completed_prescreen:
        resp = make_response(redirect(url_for('annotation')))
        resp = set_ids_cookies(resp, prolific_id, study_id, session_id)
        return resp
    else:

        resp = render_template("prescreen_v1.html", task=prescreen_task, task_type=task_type,
                               corpus_intro= corpus_intro, nav_info=get_nav_info(user, study))
        resp = make_response(resp)
        resp = set_ids_cookies(resp, prolific_id, study_id, session_id)
        return resp

@application.route('/annotation', methods=['GET', 'POST'])
def annotation():
    prolific_id, study_id, session_id, user, study = get_state_info(request, db)

    # indicator to start new task
    confirm_new_task = request.args.get('confirm', default="false") == 'true'

    if not study:
        return redirect(
            url_for('error') + '?message=' + urllib.parse.quote("Study not found or Study Id not provided!"))
    elif study["status"] != "active":
        return redirect(
            url_for('error') + '?message=' + urllib.parse.quote(f"Study(ID:{study_id}) is currently not active!"))
    task_type = study["task_type"]

    if not user:
        resp = make_response(redirect(url_for('register')))
        resp = set_ids_cookies(resp, prolific_id, study_id, session_id)
        return resp
    if not user["consent_signed_timestamp"]:
        resp = make_response(redirect(url_for('pls_page')))
        resp = set_ids_cookies(resp, prolific_id, study_id, session_id)
        return resp
    if not user[f"completed_prescreen_{task_type}"]:
        resp = make_response(redirect(url_for('prescreen')))
        resp = set_ids_cookies(resp, prolific_id, study_id, session_id)
        return resp
    if study["pre_screen_only"]:
        resp = make_response(redirect(url_for('completion')))
        resp = set_ids_cookies(resp, prolific_id, study_id, session_id)
        return resp
    if user["progress"] == 1 and not confirm_new_task:
        resp = make_response(render_template("confirm_new_task.html", nav_info=get_nav_info(user, study), study_id= study_id))
        resp = set_ids_cookies(resp, prolific_id, study_id, session_id)
        return resp
    elif user["progress"] == 1 and confirm_new_task:
        user["cur_conversation_id"] = None
        user["cur_task"] = None
        user["progress"] = 0

    ## Logic of verifying and assigning task
    cur_task_id = user.get("cur_task", None)
    cur_task_group_id = user.get("cur_conversation_id", None)

    ## confirming if study type matches with the task, if not, we will reset.
    if cur_task_id and task_type not in cur_task_id:
        remove_user_from_task(user, cur_task_id, task_type, db)
        cur_task_id = None
        cur_task_group_id = None


    if not cur_task_group_id:
        cur_task_group_id, cur_task_id, user, completion_status = assign_task_to_user(user, task_type, db)
    elif not cur_task_id:
        cur_task_group_id, cur_task_id, user, completion_status = assign_task_to_user(user, task_type, db, cur_task_group_id)

    if not cur_task_id:
        return redirect(
            url_for('error') + '?message=' + urllib.parse.quote("There is currently no task available for this study, please wait a while or contact/message the organizer."))

    original_task_id = "_".join(cur_task_id.split("_")[:-1])
    task = get_task_by_id(original_task_id, db)
    progress = user.get('progress', False) * 100

    if request.method == 'POST':
        user_annotation = request.get_json()
        valid = validate_annotation(user_annotation, task["target_utterances"], task_type)
        if valid:
            user, completed, completion_id, submission_id = submit_annotation(user_annotation, cur_task_id, task_type, user, db)
            if completion_id:
                # Render the completion page
                return jsonify({"success": True,
                                'message': f"Congratulation! You have completed all tasks in this conversation!\n (completion_id: {completion_id})",
                                'completed': completed,
                                'progress': user['progress'] * 100,
                                'submission_id': submission_id,
                                'completion_id': completion_id})
            elif user["progress"] != 1:
                return jsonify({"success": True,
                                'message': f"Task annotation submitted successfully! Few more to go! \n"
                                           f" (submission_id: {submission_id}, progress: {user['progress']*100} %)",
                                'completed': completed,
                                'progress': user['progress'] * 100,
                                'submission_id': submission_id,
                                'completion_id': completion_id})
            else:
                # Something went wrong, return JSON with an error message
                return jsonify({"success": False, 'message': "Invalid annotation format!"})
        else:
            # The input was not valid
            return jsonify({"success": False, 'message': "Invalid annotation format!"})
    nav_info = get_nav_info(user, study)
    corpus_id = task["corpus_id"]
    corpus_intro = corpus_meta[corpus_id]["intro"]
    resp = make_response(render_template("annotation_v1.html", task=task, task_type=task_type, nav_info=nav_info, progress=progress, conv_id=cur_task_group_id, corpus_intro=corpus_intro))
    resp = set_ids_cookies(resp, prolific_id, study_id, session_id)
    return resp

@application.route('/completion')
def completion():
    prolific_id, study_id, session_id, user, study = get_state_info(request, db)
    if not study:
        return redirect(
            url_for('error') + '?message=' + urllib.parse.quote("Study not found or Study Id not provided!"))
    elif study["status"] != "active":
        return redirect(
            url_for('error') + '?message=' + urllib.parse.quote(f"Study(ID:{study_id}) is currently not active!"))
    elif study["pre_screen_only"]:
        submission_id = "prescreen_" + study["task_type"] + "_" + prolific_id
    else:
        submission_id = request.args.get('submission_id')
    if submission_id:
        resp = make_response(render_template("completion.html",
                               completion_code=study["completion_code"],
                               submission_id=submission_id, nav_info=get_nav_info(user, study)))
        resp = set_ids_cookies(resp, prolific_id, study_id, session_id)
        return resp
    else:
        return redirect(
            url_for('error') + '?message=' + urllib.parse.quote(f"Requesting completion without real submission id!"))

@application.route('/check_username')
def check_username():
    user_id = request.args.get('username')
    # Replace the following with actual username existence check logic
    user = get_user_by_id(user_id, db)
    if user:
        exists = True
    else:
        exists = False
    return jsonify({'exists': exists})


@application.route('/codebook')
def codebook():
    prolific_id, study_id, session_id, user, study = get_state_info(request, db)
    if not study:
        return redirect(
            url_for('error') + '?message=' + urllib.parse.quote("Study not found or Study Id not provided!"))
    elif study["status"] != "active":
        return redirect(
            url_for('error') + '?message=' + urllib.parse.quote(f"Study(ID:{study_id}) is currently not active!"))

    task_type = study["task_type"]

    nav_info = get_nav_info(user, study)
    if task_type == "info":
        return render_template('tutorial_info.html', nav_info= nav_info, codebook=True)
    else:
        return render_template('tutorial_mix.html', nav_info= nav_info, codebook=True)

if __name__ == '__main__':
    if MODE == "DEBUG":
        application.run(debug=True, port=PORT)
    else:
        application.run(debug=False, port=PORT)