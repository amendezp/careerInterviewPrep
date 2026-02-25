import json
import os
import random
import re

from anthropic import Anthropic
from dotenv import load_dotenv
from flask import Flask, jsonify, request

load_dotenv()

app = Flask(__name__)
_client = None

def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        _client = Anthropic(api_key=api_key)
    return _client

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_study_guide():
    path = os.path.join(BASE_DIR, "data", "study_guide.md")
    with open(path, "r") as f:
        return f.read()

def load_topics():
    path = os.path.join(BASE_DIR, "data", "topics.json")
    with open(path, "r") as f:
        return json.load(f)

def extract_section(guide, topic_id):
    topics = load_topics()
    topic = next((t for t in topics if t["id"] == topic_id), None)
    if not topic:
        return guide

    start = guide.find(topic["section_start"])
    end = guide.find(topic["section_end"]) if topic.get("section_end") else len(guide)

    if start == -1:
        return guide
    if end == -1:
        end = len(guide)

    return guide[start:end].strip()

def parse_json_response(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


@app.route("/api/topics", methods=["GET"])
def get_topics():
    topics = load_topics()
    return jsonify([{"id": t["id"], "name": t["name"]} for t in topics])


@app.route("/api/question", methods=["POST"])
def generate_question():
    data = request.get_json()
    topic = data.get("topic", "all")
    fmt = data.get("format", "random")
    difficulty = data.get("difficulty", "intermediate")
    history = data.get("history", [])
    topic_counts = data.get("topic_counts", {})

    guide = load_study_guide()
    topics_list = load_topics()

    if topic == "all":
        if topic_counts:
            # Weight selection toward least-attempted topics for balanced coverage
            max_count = max(topic_counts.values(), default=0) + 1
            weights = [max_count - topic_counts.get(t["id"], 0) for t in topics_list]
            chosen = random.choices(topics_list, weights=weights, k=1)[0]
        else:
            chosen = random.choice(topics_list)
        topic = chosen["id"]

    context = extract_section(guide, topic)

    if fmt == "random":
        fmt = random.choice(["multiple_choice", "open_ended", "scenario"])

    if fmt == "multiple_choice":
        fmt_instruction = "Generate a multiple_choice question with exactly 4 options (A, B, C, D). Wrong answers should represent real misconceptions, not obviously silly options."
    elif fmt == "open_ended":
        fmt_instruction = "Generate an open_ended question that requires a 2-4 sentence explanation of concepts or trade-offs."
    else:
        fmt_instruction = "Generate a scenario question presenting a realistic architectural decision with genuine trade-offs."

    difficulty_instruction = {
        "beginner": "Test basic understanding — definitions, simple use cases, straightforward comparisons. A person who read the material once should be able to answer.",
        "intermediate": "Test applied understanding — when to use what, trade-offs, common pitfalls, practical recommendations. Requires having internalized the material.",
        "advanced": "Test deep understanding — edge cases, combining concepts, challenging misconceptions, nuanced architectural decisions. The kind of question that separates someone who memorized from someone who truly understands."
    }.get(difficulty, "intermediate")

    history_text = ""
    if history:
        history_text = f"\n\nPrevious questions asked (DO NOT repeat these or ask very similar questions):\n" + "\n".join(f"- {h}" for h in history[-20:])

    prompt = f"""You are a technical interviewer preparing questions about LLM/GenAI topics for a Stanford technical screening.

STUDY MATERIAL:
{context}

INSTRUCTIONS:
- {fmt_instruction}
- Difficulty: {difficulty} — {difficulty_instruction}
- Questions should test understanding "one level deeper than textbook" — the kind of depth a senior practitioner would expect.
- You should supplement the study material with your OWN broader knowledge of the topic — include real-world patterns, production war stories, recent developments, and cross-cutting concepts that a senior practitioner would know.
- Do NOT limit yourself to rephrasing the study material. Use it as a foundation but generate questions that test deeper, more varied understanding.
- For multiple choice: wrong answers should be plausible misconceptions, not jokes.
- For open-ended: the answer should require explaining WHY, not just WHAT.
- For scenarios: present a realistic situation where the candidate must recommend an approach and justify it.
- NEVER use the name "Eli" in any question or answer. If referencing an interviewer, say "the interviewer".
{history_text}

Respond with ONLY valid JSON in this exact format:

For multiple_choice:
{{
  "type": "multiple_choice",
  "topic": "{topic}",
  "difficulty": "{difficulty}",
  "question": "the question text",
  "options": {{
    "A": "option A text",
    "B": "option B text",
    "C": "option C text",
    "D": "option D text"
  }},
  "correct_answer": "A",
  "explanation": "brief explanation of why the correct answer is right and why the others are wrong",
  "summary": "5-8 word summary of what this question tests"
}}

For open_ended:
{{
  "type": "open_ended",
  "topic": "{topic}",
  "difficulty": "{difficulty}",
  "question": "the question text",
  "key_points": ["point 1 a good answer should mention", "point 2", "point 3"],
  "sample_answer": "a strong 2-4 sentence answer",
  "summary": "5-8 word summary of what this question tests"
}}

For scenario:
{{
  "type": "scenario",
  "topic": "{topic}",
  "difficulty": "{difficulty}",
  "question": "the scenario description and question",
  "key_points": ["point 1 a good answer should cover", "point 2", "point 3"],
  "sample_answer": "a strong response covering the key trade-offs",
  "summary": "5-8 word summary of what this question tests"
}}"""

    try:
        response = get_client().messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        result = parse_json_response(response.content[0].text)
        if result:
            return jsonify(result)
        else:
            return jsonify({"error": "Failed to parse question response"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/evaluate", methods=["POST"])
def evaluate_answer():
    data = request.get_json()
    question_data = data.get("question", {})
    user_answer = data.get("answer", "")

    guide = load_study_guide()
    topic = question_data.get("topic", "all")
    context = extract_section(guide, topic) if topic != "all" else guide

    q_type = question_data.get("type", "open_ended")

    if q_type == "multiple_choice":
        eval_prompt = f"""You are evaluating a quiz answer about LLM/GenAI topics.

STUDY MATERIAL (for reference):
{context}

QUESTION: {question_data.get('question', '')}

OPTIONS:
{json.dumps(question_data.get('options', {}), indent=2)}

CORRECT ANSWER: {question_data.get('correct_answer', '')}

USER'S ANSWER: {user_answer}

IMPORTANT: Never use the name "Eli" — refer to "the interviewer" instead.

Evaluate the user's answer. Respond with ONLY valid JSON:
{{
  "correct": true/false,
  "score": 1.0 or 0.0,
  "strengths": ["What the user got right (1-2 bullet points, or empty array if completely wrong)"],
  "gaps": ["What was wrong or missing (1-2 bullet points, or empty array if perfect)"],
  "explanation": "Full explanation of the correct answer with context from the study material",
  "tip": "A practical study tip or interview insight related to this question"
}}"""
    else:
        eval_prompt = f"""You are evaluating a quiz answer about LLM/GenAI topics.

STUDY MATERIAL (for reference):
{context}

QUESTION: {question_data.get('question', '')}

KEY POINTS A GOOD ANSWER SHOULD COVER:
{json.dumps(question_data.get('key_points', []), indent=2)}

SAMPLE STRONG ANSWER:
{question_data.get('sample_answer', '')}

USER'S ANSWER: {user_answer}

IMPORTANT: Never use the name "Eli" — refer to "the interviewer" instead.

Evaluate how well the user's answer covers the key concepts. Be encouraging but honest. Respond with ONLY valid JSON:
{{
  "correct": true/false (true if they demonstrate solid understanding, even if not perfect),
  "score": 0.0-1.0 (0.0 = completely wrong, 0.5 = partial understanding, 1.0 = excellent),
  "strengths": ["Specific things the user got right (1-3 bullet points). Reference concepts from the study material. Empty array if completely wrong."],
  "gaps": ["Specific things the user missed or got wrong (1-3 bullet points). Reference concepts from the study material. Empty array if perfect."],
  "explanation": "The complete correct answer with full context from the study material",
  "tip": "A practical study tip or interview insight related to this question"
}}"""

    try:
        response = get_client().messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[{"role": "user", "content": eval_prompt}],
        )

        result = parse_json_response(response.content[0].text)
        if result:
            return jsonify(result)
        else:
            return jsonify({"error": "Failed to parse evaluation response"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/recommendations", methods=["POST"])
def get_recommendations():
    data = request.get_json()
    progress = data.get("progress", {})

    if not progress:
        return jsonify({"error": "No progress data provided"}), 400

    topics_list = load_topics()
    topic_map = {t["id"]: t["name"] for t in topics_list}

    summary_lines = []
    for tid, name in topic_map.items():
        p = progress.get(tid)
        if p and p.get("attempted", 0) > 0:
            avg = round(p["totalScore"] / p["attempted"] * 100)
            summary_lines.append(f"- {name}: {p['attempted']} questions, {avg}% avg score")
        else:
            summary_lines.append(f"- {name}: not attempted")

    progress_summary = "\n".join(summary_lines)

    guide = load_study_guide()

    prompt = f"""You are a study coach helping someone prepare for a Stanford technical screening on LLM/GenAI topics.

STUDY MATERIAL:
{guide[:3000]}

STUDENT'S PROGRESS:
{progress_summary}

Based on their progress, give specific, actionable study advice in 4-6 sentences. Focus on:
1. Their weakest areas and what concepts to review
2. Topics they haven't attempted yet that are important
3. Concrete study strategies (not generic advice)

Be direct and specific. Reference actual concepts from the study material."""

    try:
        response = get_client().messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )

        advice = response.content[0].text
        return jsonify({"advice": advice})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
