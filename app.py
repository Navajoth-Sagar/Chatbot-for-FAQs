from flask import Flask, render_template, request
import chatbot

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    response = ""
    if request.method == "POST":
        user_msg = request.form["message"]
        response = chatbot.chatbot_response(user_msg)
    return render_template("index.html", bot_reply=response)

if __name__ == "__main__":
    app.run(debug=True)
