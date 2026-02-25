// Topic importance weights for recommendation scoring
const TOPIC_IMPORTANCE = {
  rag: 1.0,
  fine_tuning: 1.0,
  prompt_engineering: 1.0,
  decision_framework: 1.0,
  agents: 1.0,
  evals: 0.6,
  embeddings: 0.6,
  cot: 0.6,
  memory: 0.6,
  lora: 0.3,
  guardrails: 0.3,
};

const RECOMMENDATION_THRESHOLD = 10;

// State
const STATE_KEY = "quiz_progress";

let topics = [];
let currentTopic = "all";
let currentQuestion = null;
let selectedMcOption = null;

// DOM elements
const topicBar = document.getElementById("topic-bar");
const formatSelect = document.getElementById("format-select");
const difficultySelect = document.getElementById("difficulty-select");
const newQuestionBtn = document.getElementById("new-question-btn");
const questionArea = document.getElementById("question-area");
const questionTopic = document.getElementById("question-topic");
const questionType = document.getElementById("question-type");
const questionDifficulty = document.getElementById("question-difficulty");
const questionText = document.getElementById("question-text");
const mcOptions = document.getElementById("mc-options");
const textAnswer = document.getElementById("text-answer");
const answerInput = document.getElementById("answer-input");
const submitBtn = document.getElementById("submit-btn");
const loadingEl = document.getElementById("loading");
const feedbackArea = document.getElementById("feedback-area");
const feedbackIcon = document.getElementById("feedback-icon");
const feedbackLabel = document.getElementById("feedback-label");
const feedbackScore = document.getElementById("feedback-score");
const feedbackText = document.getElementById("feedback-text");
const explanationText = document.getElementById("explanation-text");
const tipText = document.getElementById("tip-text");
const nextBtn = document.getElementById("next-btn");
const progressGrid = document.getElementById("progress-grid");
const progressSummary = document.getElementById("progress-summary");
const resetBtn = document.getElementById("reset-btn");
const emptyState = document.getElementById("empty-state");
const recommendationsSection = document.getElementById("recommendations");
const recommendationsSubtitle = document.getElementById("recommendations-subtitle");
const recommendationsList = document.getElementById("recommendations-list");
const deepAnalysisBtn = document.getElementById("deep-analysis-btn");
const deepAnalysisResult = document.getElementById("deep-analysis-result");

// Progress persistence
function getProgress() {
  try {
    return JSON.parse(localStorage.getItem(STATE_KEY)) || {};
  } catch {
    return {};
  }
}

function saveProgress(progress) {
  localStorage.setItem(STATE_KEY, JSON.stringify(progress));
}

function getHistory() {
  try {
    return JSON.parse(localStorage.getItem("quiz_history")) || [];
  } catch {
    return [];
  }
}

function addToHistory(summary) {
  const history = getHistory();
  history.push(summary);
  if (history.length > 50) history.shift();
  localStorage.setItem("quiz_history", JSON.stringify(history));
}

function recordAnswer(topicId, score) {
  const progress = getProgress();
  if (!progress[topicId]) {
    progress[topicId] = { attempted: 0, totalScore: 0 };
  }
  progress[topicId].attempted++;
  progress[topicId].totalScore += score;
  saveProgress(progress);
  renderProgress();
}

// API calls
async function fetchTopics() {
  const res = await fetch("/api/topics");
  return res.json();
}

async function fetchQuestion(topic, format, difficulty, history) {
  const res = await fetch("/api/question", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ topic, format, difficulty, history }),
  });
  if (!res.ok) throw new Error("Failed to generate question");
  return res.json();
}

async function evaluateAnswer(question, answer) {
  const res = await fetch("/api/evaluate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, answer }),
  });
  if (!res.ok) throw new Error("Failed to evaluate answer");
  return res.json();
}

// UI rendering
function renderTopicButtons() {
  // "All Topics" button already in HTML
  topics.forEach((t) => {
    const btn = document.createElement("button");
    btn.className = "topic-btn";
    btn.dataset.topic = t.id;
    btn.textContent = t.name;
    topicBar.appendChild(btn);
  });
}

function setActiveTopic(topicId) {
  currentTopic = topicId;
  document.querySelectorAll(".topic-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.topic === topicId);
  });
}

function showSection(sectionId) {
  ["question-area", "loading", "feedback-area", "empty-state"].forEach((id) => {
    document.getElementById(id).classList.toggle("hidden", id !== sectionId);
  });
}

function renderQuestion(q) {
  currentQuestion = q;
  selectedMcOption = null;

  // Meta badges
  const topicObj = topics.find((t) => t.id === q.topic);
  questionTopic.textContent = topicObj ? topicObj.name : q.topic;
  questionType.textContent = q.type.replace("_", " ");
  questionDifficulty.textContent = q.difficulty;
  questionDifficulty.className = "badge " + q.difficulty;

  questionText.textContent = q.question;

  // Show appropriate answer UI
  if (q.type === "multiple_choice") {
    mcOptions.classList.remove("hidden");
    textAnswer.classList.add("hidden");

    document.querySelectorAll(".mc-btn").forEach((btn) => {
      const opt = btn.dataset.option;
      btn.querySelector(".mc-text").textContent = q.options[opt];
      btn.classList.remove("selected", "correct", "incorrect");
      btn.disabled = false;
    });
  } else {
    mcOptions.classList.add("hidden");
    textAnswer.classList.remove("hidden");
    answerInput.value = "";
  }

  submitBtn.classList.remove("hidden");
  submitBtn.disabled = q.type === "multiple_choice"; // need to select an option first

  showSection("question-area");
}

function renderFeedback(result) {
  const score = result.score;

  if (score >= 0.8) {
    feedbackIcon.textContent = "\u2713";
    feedbackLabel.textContent = "Correct!";
    feedbackLabel.className = "correct";
  } else if (score >= 0.4) {
    feedbackIcon.textContent = "~";
    feedbackLabel.textContent = "Partial";
    feedbackLabel.className = "partial";
  } else {
    feedbackIcon.textContent = "\u2717";
    feedbackLabel.textContent = "Incorrect";
    feedbackLabel.className = "incorrect";
  }

  feedbackScore.textContent = `Score: ${Math.round(score * 100)}%`;
  feedbackText.textContent = result.feedback;
  explanationText.textContent = result.explanation;
  tipText.textContent = result.tip;

  showSection("feedback-area");
  // Keep question visible above feedback
  questionArea.classList.remove("hidden");
  submitBtn.classList.add("hidden");
}

function computeRecommendations() {
  const progress = getProgress();

  // Count total answers across all topics
  let totalAnswers = 0;
  for (const data of Object.values(progress)) {
    totalAnswers += data.attempted || 0;
  }

  if (totalAnswers < RECOMMENDATION_THRESHOLD) return null;

  // Score each topic
  const scored = topics.map((t) => {
    const data = progress[t.id] || { attempted: 0, totalScore: 0 };
    const importance = TOPIC_IMPORTANCE[t.id] || 0.5;

    let gapPenalty, weaknessPenalty;

    if (data.attempted === 0) {
      // Never attempted — high gap, assume weakness
      gapPenalty = 1.0;
      weaknessPenalty = 0.8;
    } else {
      gapPenalty = 0.0;
      const avgScore = data.totalScore / data.attempted;
      weaknessPenalty = 1 - avgScore;

      // Boost undertested topics (fewer than 3 questions)
      if (data.attempted < 3) {
        gapPenalty = 0.4;
      }
    }

    const priorityScore =
      gapPenalty * 0.3 + weaknessPenalty * 0.5 + importance * 0.2;

    // Build reason string
    let reason;
    if (data.attempted === 0) {
      reason = "Not attempted yet";
    } else if (data.attempted < 3) {
      const pct = Math.round((data.totalScore / data.attempted) * 100);
      reason = `Only ${data.attempted} question${data.attempted > 1 ? "s" : ""} answered (${pct}% avg)`;
    } else {
      const pct = Math.round((data.totalScore / data.attempted) * 100);
      reason = `${data.attempted} questions, ${pct}% avg score`;
    }

    return {
      topicId: t.id,
      topicName: t.name,
      priorityScore,
      reason,
      attempted: data.attempted,
      avgScore: data.attempted > 0 ? data.totalScore / data.attempted : 0,
    };
  });

  // Sort by priority descending, return top 3
  scored.sort((a, b) => b.priorityScore - a.priorityScore);
  return { recommendations: scored.slice(0, 3), totalAnswers };
}

function renderRecommendations() {
  const result = computeRecommendations();

  if (!result) {
    recommendationsSection.classList.add("hidden");
    return;
  }

  const { recommendations, totalAnswers } = result;
  recommendationsSection.classList.remove("hidden");
  recommendationsSubtitle.textContent = `Based on ${totalAnswers} answers — here's where to focus next`;

  recommendationsList.innerHTML = "";
  recommendations.forEach((rec, i) => {
    const rank = i + 1;
    const rankClass = `rank-${rank}`;
    const scorePct = Math.round(rec.priorityScore * 100);

    const card = document.createElement("div");
    card.className = "rec-card";
    card.innerHTML = `
      <div class="rec-card-header">
        <span class="rec-rank ${rankClass}">#${rank}</span>
        <span class="rec-topic-name">${rec.topicName}</span>
      </div>
      <div class="rec-reason">${rec.reason}</div>
      <div class="rec-score-bar">
        <div class="rec-score-bar-fill" style="width: ${scorePct}%"></div>
      </div>
      <button class="rec-practice-btn" data-topic="${rec.topicId}">Practice This Topic</button>
    `;
    recommendationsList.appendChild(card);
  });

  // Reset deep analysis when recommendations update
  deepAnalysisResult.classList.add("hidden");
  deepAnalysisResult.textContent = "";
}

function renderProgress() {
  const progress = getProgress();
  progressGrid.innerHTML = "";

  let totalAttempted = 0;
  let totalScore = 0;

  topics.forEach((t) => {
    const data = progress[t.id] || { attempted: 0, totalScore: 0 };
    totalAttempted += data.attempted;
    totalScore += data.totalScore;

    if (data.attempted === 0) return;

    const avg = data.totalScore / data.attempted;
    const pct = Math.round(avg * 100);
    const colorClass = pct >= 70 ? "green" : pct >= 40 ? "yellow" : "red";

    const card = document.createElement("div");
    card.className = "progress-card";
    card.innerHTML = `
      <div class="topic-name">${t.name}</div>
      <div class="stats">${data.attempted} answered &middot; ${pct}% avg</div>
      <div class="progress-bar">
        <div class="progress-bar-fill ${colorClass}" style="width: ${pct}%"></div>
      </div>
    `;
    progressGrid.appendChild(card);
  });

  if (totalAttempted > 0) {
    const overallPct = Math.round((totalScore / totalAttempted) * 100);
    progressSummary.textContent = `${totalAttempted} questions answered \u00B7 ${overallPct}% overall`;
  } else {
    progressSummary.textContent = "No questions answered yet.";
  }

  renderRecommendations();
}

// Event handlers
topicBar.addEventListener("click", (e) => {
  const btn = e.target.closest(".topic-btn");
  if (!btn) return;
  setActiveTopic(btn.dataset.topic);
});

newQuestionBtn.addEventListener("click", async () => {
  const format = formatSelect.value;
  const difficulty = difficultySelect.value;

  newQuestionBtn.disabled = true;
  showSection("loading");
  loadingEl.querySelector("p").textContent = "Generating question...";

  try {
    const history = getHistory();
    const q = await fetchQuestion(currentTopic, format, difficulty, history);
    if (q.error) throw new Error(q.error);
    renderQuestion(q);
  } catch (err) {
    showSection("empty-state");
    document.getElementById("empty-state").querySelector("p").textContent =
      "Error: " + err.message + ". Check your API key and try again.";
  } finally {
    newQuestionBtn.disabled = false;
  }
});

mcOptions.addEventListener("click", (e) => {
  const btn = e.target.closest(".mc-btn");
  if (!btn || btn.disabled) return;

  document.querySelectorAll(".mc-btn").forEach((b) => b.classList.remove("selected"));
  btn.classList.add("selected");
  selectedMcOption = btn.dataset.option;
  submitBtn.disabled = false;
});

submitBtn.addEventListener("click", async () => {
  if (!currentQuestion) return;

  let answer;
  if (currentQuestion.type === "multiple_choice") {
    if (!selectedMcOption) return;
    answer = selectedMcOption;
  } else {
    answer = answerInput.value.trim();
    if (!answer) return;
  }

  submitBtn.disabled = true;
  submitBtn.textContent = "Evaluating...";

  // Disable MC buttons
  if (currentQuestion.type === "multiple_choice") {
    document.querySelectorAll(".mc-btn").forEach((btn) => {
      btn.disabled = true;
    });
  }

  try {
    const result = await evaluateAnswer(currentQuestion, answer);
    if (result.error) throw new Error(result.error);

    // Highlight MC answers
    if (currentQuestion.type === "multiple_choice") {
      const correctLetter = currentQuestion.correct_answer;
      document.querySelectorAll(".mc-btn").forEach((btn) => {
        if (btn.dataset.option === correctLetter) {
          btn.classList.add("correct");
        } else if (btn.dataset.option === selectedMcOption && selectedMcOption !== correctLetter) {
          btn.classList.add("incorrect");
        }
      });
    }

    // Record progress
    const topicId = currentQuestion.topic !== "all" ? currentQuestion.topic : topics[0]?.id || "all";
    recordAnswer(topicId, result.score);
    addToHistory(currentQuestion.summary || currentQuestion.question.slice(0, 60));

    renderFeedback(result);
  } catch (err) {
    submitBtn.textContent = "Error - Try Again";
    submitBtn.disabled = false;
  }
});

nextBtn.addEventListener("click", () => {
  newQuestionBtn.click();
});

resetBtn.addEventListener("click", () => {
  if (confirm("Reset all progress? This cannot be undone.")) {
    localStorage.removeItem(STATE_KEY);
    localStorage.removeItem("quiz_history");
    renderProgress();
  }
});

// Practice This Topic buttons (delegated)
recommendationsList.addEventListener("click", (e) => {
  const btn = e.target.closest(".rec-practice-btn");
  if (!btn) return;
  const topicId = btn.dataset.topic;
  setActiveTopic(topicId);
  window.scrollTo({ top: 0, behavior: "smooth" });
  newQuestionBtn.click();
});

// Get Study Advice from Claude
deepAnalysisBtn.addEventListener("click", async () => {
  const progress = getProgress();
  deepAnalysisBtn.disabled = true;
  deepAnalysisBtn.textContent = "Getting advice...";

  try {
    const res = await fetch("/api/recommendations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ progress }),
    });
    if (!res.ok) throw new Error("Failed to get recommendations");
    const data = await res.json();

    deepAnalysisResult.textContent = data.advice;
    deepAnalysisResult.classList.remove("hidden");
  } catch (err) {
    deepAnalysisResult.textContent = "Error: " + err.message;
    deepAnalysisResult.classList.remove("hidden");
  } finally {
    deepAnalysisBtn.disabled = false;
    deepAnalysisBtn.textContent = "Get Study Advice from Claude";
  }
});

// Keyboard shortcut: Enter to submit
document.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    if (!submitBtn.classList.contains("hidden") && !submitBtn.disabled) {
      // Only for MC or if textarea is not focused
      if (currentQuestion?.type === "multiple_choice" || document.activeElement !== answerInput) {
        submitBtn.click();
      }
    }
  }
});

// MC keyboard shortcuts (A, B, C, D)
document.addEventListener("keydown", (e) => {
  if (!currentQuestion || currentQuestion.type !== "multiple_choice") return;
  if (submitBtn.classList.contains("hidden")) return;

  const key = e.key.toUpperCase();
  if (["A", "B", "C", "D"].includes(key)) {
    const btn = document.querySelector(`.mc-btn[data-option="${key}"]`);
    if (btn && !btn.disabled) btn.click();
  }
});

// Init
async function init() {
  try {
    topics = await fetchTopics();
    renderTopicButtons();
    renderProgress();
    showSection("empty-state");
  } catch (err) {
    document.getElementById("empty-state").querySelector("p").textContent =
      "Failed to load topics. Make sure the API is running.";
  }
}

init();
