from flask import Flask, render_template, request, jsonify, session, flash, url_for, redirect
from pymongo import MongoClient
import os
from flask_bcrypt import Bcrypt
from datetime import datetime
from dotenv import load_dotenv
import uuid
from bson import ObjectId
import threading
from RAG.archive_pipeline import archive_session

load_dotenv()
MONGO_URI = os.environ.get("MONGO_URI")

app = Flask(__name__)
db_client = MongoClient(MONGO_URI)
db = db_client["TechNova"]
users = db["farmers"]
conversations = db["conversations"]
bcrypt = Bcrypt(app)
app.secret_key = os.environ.get("SECRET_KEY")

# Lazy-loaded graphs
_compiled_graph = None
_compiled_fast_graph = None
_graph_lock = threading.Lock()

def get_graphs():
    global _compiled_graph, _compiled_fast_graph
    if _compiled_graph is None:
        with _graph_lock:
            if _compiled_graph is None:  # double-checked locking
                from utils.AudioGraph import graph, fast_graph
                _compiled_graph = graph.compile()
                _compiled_fast_graph = fast_graph.compile()
    return _compiled_graph, _compiled_fast_graph


def run_rag(user_id):
    try:
        chat_cursor = conversations.find({
            "user_id": user_id,
            "Isarchieved": False
        })
        chat_list = list(chat_cursor)
        if not chat_list:
            return
        archive_session(user_id=user_id, messages=chat_list)
        conversations.update_many(
            {"user_id": user_id, "Isarchieved": False},
            {"$set": {"Isarchieved": True}}
        )
    except Exception as e:
        print(f"[RAG ERROR] {str(e)}")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        if not email or not password:
            flash("Credentials are missing")
            return redirect(url_for("login"))

        user = users.find_one({"email": email})
        if user:
            stored_password = user.get("password_hash")
            if stored_password and bcrypt.check_password_hash(stored_password, password):
                session["user"] = str(user["_id"])
                session["session_id"] = str(uuid.uuid4())
                thread = threading.Thread(
                    target=run_rag,
                    args=(ObjectId(session["user"]),)
                )
                thread.start()
                flash("Login Successful!")
                return redirect(url_for("user_dashboard"))
            flash("Invalid password!")
            return redirect(url_for("login"))
        flash("Invalid Email")
        return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        data = request.form
        name = data.get("name")
        email = data.get("email")
        password = data.get("password")
        location = data.get("location")
        farm_size = data.get("farm_size")
        soil_type = data.get("soil_type")
        water_availability = data.get("water_availabilty")
        prev_crop = data.get("prev_crop")
        phone = data.get("phone", None)

        if not name or not email or not password:
            flash("Credentials missing")
            return redirect(url_for("signup"))

        user = users.find_one({"email": email})
        if user:
            flash("User already exists with this mail")
            return redirect(url_for("signup"))

        password_hash = bcrypt.generate_password_hash(password).decode("utf-8")
        new_user = {
            "name": name,
            "email": email,
            "password_hash": password_hash,
            "phone": phone,
            "location": location,
            "farm_size": farm_size,
            "soil_type": soil_type,
            "water_availability": water_availability,
            "previous_crop": prev_crop,
            "role": "user",
            "created_at": datetime.utcnow()
        }
        users.insert_one(new_user)
        return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/audio_graph", methods=["POST"])
def audio_graph():
    user_id = session.get("user")
    session_id = session.get("session_id")
    if not user_id or not session_id:
        return jsonify({"error": "Unauthorized"}), 401

    user_query = request.json.get("user_query")
    conversations.update_one(
        {"session_id": session_id},
        {
            "$setOnInsert": {
                "user_id": ObjectId(user_id),
                "session_id": session_id,
                "created_at": datetime.utcnow(),
                "Isarchieved": False
            },
            "$push": {
                "messages": {
                    "role": "user",
                    "content": user_query,
                    "timestamp": datetime.utcnow()
                }
            }
        },
        upsert=True
    )

    chat_session = conversations.find_one({"session_id": session_id})
    user = users.find_one({"_id": ObjectId(user_id)})
    userName = user["name"]
    userLoc = user["location"]

    compiled_graph, _ = get_graphs()
    output_state = compiled_graph.invoke({
        "user_query": user_query,
        "user_id": str(user_id),
        "user_name": userName,
        "user_location": userLoc,
        "chat_history": chat_session.get("messages", [])
    })

    assistant_output = output_state["output_text"]
    conversations.update_one(
        {"session_id": session_id},
        {"$push": {"messages": {"role": "assistant", "content": assistant_output}}}
    )
    return jsonify({"response": assistant_output})


@app.route("/fast_graph", methods=["POST"])
def text_graph():
    user_id = session.get("user")
    session_id = session.get("session_id")
    if not user_id or not session_id:
        return jsonify({"error": "Unauthorized"}), 401

    user_query = request.json.get("user_query")
    conversations.update_one(
        {"session_id": session_id},
        {
            "$setOnInsert": {
                "user_id": ObjectId(user_id),
                "session_id": session_id,
                "created_at": datetime.utcnow(),
                "Isarchieved": False
            },
            "$push": {
                "messages": {
                    "role": "user",
                    "content": user_query,
                    "timestamp": datetime.utcnow()
                }
            }
        },
        upsert=True
    )

    chat_session = conversations.find_one({"session_id": session_id})
    user = users.find_one({"_id": ObjectId(user_id)})
    userName = user["name"]
    userLoc = user["location"]

    _, compiled_fast_graph = get_graphs()
    output_state = compiled_fast_graph.invoke({
        "user_query": user_query,
        "user_id": str(user_id),
        "user_name": userName,
        "user_location": userLoc,
        "chat_history": chat_session.get("messages", [])
    })

    assistant_output = output_state["output_text"]
    conversations.update_one(
        {"session_id": session_id},
        {"$push": {"messages": {"role": "assistant", "content": assistant_output}}}
    )
    return jsonify({"response": assistant_output})


@app.route("/dashboard")
def user_dashboard():
    if "user" not in session:
        flash("Please log in first.")
        return redirect(url_for("login"))
    return render_template("user_dashboard.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)