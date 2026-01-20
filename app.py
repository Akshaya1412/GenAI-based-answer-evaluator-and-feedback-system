from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch

torch.set_num_threads(1)

app = Flask(__name__)

# Load models
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
feedback_model = pipeline("text-generation", model="gpt2")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        question = request.form["question"]
        model_answer = request.form["model_answer"]
        student_answer = request.form["student_answer"]
        max_marks = int(request.form["max_marks"])

        embeddings = similarity_model.encode(
            [model_answer, student_answer],
            convert_to_tensor=True
        )

        similarity_score = util.cos_sim(embeddings[0], embeddings[1]).item()
        marks = round(similarity_score * max_marks, 2)

        prompt = f"""
        Question: {question}
        Model Answer: {model_answer}
        Student Answer: {student_answer}
        Provide short, constructive feedback.
        """

        raw_output = feedback_model(
            prompt,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            truncation=True
        )[0]["generated_text"]

        feedback = raw_output.replace(prompt, "").strip()

        result = {
            "similarity": round(similarity_score, 2),
            "marks": marks,
            "feedback": feedback
        }

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)


