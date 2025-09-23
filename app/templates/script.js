async function askQuestion() {
  const q = document.getElementById("question").value;
  if (!q) return;
  document.getElementById("answer").innerText = "Thinking...";
  const res = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: q }),
  });
  const data = await res.json();
  if (data.error) {
    document.getElementById("answer").innerText = "Error: " + data.error;
    return;
  }
  document.getElementById("answer").innerText = data.answer || "No answer";
  const srcElem = document.getElementById("sources");
  srcElem.innerHTML = "";
  (data.sources || []).forEach((s) => {
    const div = document.createElement("div");
    div.className = "source";
    div.innerHTML = `<strong>${s.source}</strong><div>${s.snippet}</div>`;
    srcElem.appendChild(div);
  });
}

document.getElementById("askBtn").addEventListener("click", askQuestion);
document.getElementById("question").addEventListener("keydown", function (e) {
  if (e.key === "Enter") askQuestion();
});

document.getElementById("uploadBtn").addEventListener("click", async () => {
  const fileInput = document.getElementById("pdfUpload");
  if (fileInput.files.length === 0) return;
  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  const res = await fetch("/upload", {
    method: "POST",
    body: formData,
  });
  const data = await res.json();
  alert(data.success || data.error);
});

function updateChatHistory() {
  const chatDiv = document.getElementById("chatHistory");
  chatDiv.innerHTML = "";
  const history = window.chatHistory || [];
  history.forEach((item) => {
    const q = document.createElement("div");
    q.className = "question";
    q.innerText = "Q: " + item.question;
    const a = document.createElement("div");
    a.className = "answer";
    a.innerText = "A: " + item.answer;
    chatDiv.appendChild(q);
    chatDiv.appendChild(a);
  });
}

// call updateChatHistory() on page load
updateChatHistory();
