const API_URL = "/agent/chat";

const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");
const messageList = document.getElementById("messageList");
const clearBtn = document.getElementById("clearBtn");
const sendBtn = document.getElementById("sendBtn");
const messageTemplate = document.getElementById("messageTemplate");
const quickButtons = document.querySelectorAll(".quick-btn");

function createMessage(roleLabel, content, className = "") {
  const node = messageTemplate.content.firstElementChild.cloneNode(true);
  node.classList.add(className);
  node.querySelector(".message-role").textContent = roleLabel;
  node.querySelector(".message-content").innerHTML = content;
  return node;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderAssistantPayload(data) {
  const answer = escapeHtml(data.answer || "无返回结果");
  const toolUsed = escapeHtml(data.tool_used || "unknown");
  const sources = Array.isArray(data.sources) ? data.sources : [];

  let html = `<div>${answer}</div>`;

  // 工具区：始终显示，但更紧凑
  html += `
    <div class="meta-block compact-block">
      <div class="meta-title">工具</div>
      <div><code>${toolUsed}</code></div>
    </div>
  `;

  // Sources 只有非空时才显示
  if (sources.length > 0) {
    html += `<div class="meta-block compact-block"><div class="meta-title">Sources</div>`;

    html += sources.map((source, index) => {
      const file = escapeHtml(source.file || `source_${index + 1}`);
      const content = escapeHtml((source.content || "").slice(0, 180));
      return `
        <div class="source-item">
          <div><strong>${file}</strong></div>
          <div>${content}${source.content && source.content.length > 180 ? "..." : ""}</div>
        </div>
      `;
    }).join("");

    html += `</div>`;
  }

  return html;
}

function appendMessage(node) {
  messageList.appendChild(node);
  messageList.scrollTop = messageList.scrollHeight;
}

function setLoadingState(loading) {
  sendBtn.disabled = loading;
  questionInput.disabled = loading;
  sendBtn.textContent = loading ? "发送中..." : "发送";
}

async function sendQuestion(question) {
  const userText = question.trim();
  if (!userText) return;

  appendMessage(createMessage("用户", escapeHtml(userText), "user"));

  const loadingNode = createMessage("系统", "正在调用 FastAPI /agent/chat", "system-note");
  loadingNode.querySelector(".message-content").classList.add("loading-dot");
  appendMessage(loadingNode);

  setLoadingState(true);

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: userText })
    });

    const data = await response.json().catch(() => ({}));
    loadingNode.remove();

    if (!response.ok) {
      const errText = escapeHtml(data.detail || data.answer || "接口调用失败");
      appendMessage(createMessage("系统错误", errText, "error"));
      return;
    }

    appendMessage(createMessage("助手", renderAssistantPayload(data), "assistant"));
  } catch (error) {
    loadingNode.remove();
    appendMessage(createMessage("系统错误", escapeHtml(error.message || "网络请求失败"), "error"));
  } finally {
    setLoadingState(false);
    questionInput.focus();
  }
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  await sendQuestion(questionInput.value);
  questionInput.value = "";
});

clearBtn.addEventListener("click", () => {
  messageList.innerHTML = "";
  appendMessage(createMessage("系统", "对话已清空。你可以重新提问。", "system-note"));
});

quickButtons.forEach((btn) => {
  btn.addEventListener("click", async () => {
    const question = btn.dataset.question || "";
    questionInput.value = question;
    await sendQuestion(question);
    questionInput.value = "";
  });
});
