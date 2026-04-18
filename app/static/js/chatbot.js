// ====================== CHATBOT.JS - PREMIUM VERSION ======================

const chat = {
  toggle: document.getElementById("chat-toggle"),
  window: document.getElementById("chat-window"),
  messages: document.getElementById("chat-messages"),
  input: document.getElementById("chat-input"),
  send: document.getElementById("chat-send"),
  close: document.getElementById("chat-close"),
};

let lastSender = null;

// Toggle chatbot
if (chat.toggle && chat.window) {
  chat.toggle.addEventListener("click", () => {
    chat.window.classList.toggle("hidden");
    if (!chat.window.classList.contains("hidden")) {
      chat.input.focus();
      smoothScrollToBottom();
    }
  });
}

if (chat.close) {
  chat.close.addEventListener("click", () => {
    chat.window.classList.add("hidden");
  });
}

// Smooth scroll function
function smoothScrollToBottom() {
  setTimeout(() => {
    chat.messages.scrollTo({
      top: chat.messages.scrollHeight,
      behavior: "smooth"
    });
  }, 50); // Slight delay to ensure content is rendered
}

// Thêm tin nhắn (hỗ trợ markdown cho bot)
function addChatMessage(text, isUser = false) {
  const currentSender = isUser ? "user" : "bot";
  
  // Row container
  const row = document.createElement("div");
  row.className = `chat-row ${isUser ? "user-row" : "bot-row"}`;
  
  // Smart Spacing: Add 'new-sender' class if sender changed
  if (lastSender !== currentSender) {
    row.classList.add("new-sender");
  }
  lastSender = currentSender;

  // Avatar
  const avatar = document.createElement("div");
  avatar.className = "chat-avatar";
  avatar.innerHTML = isUser ? '<i class="fas fa-user"></i>' : "🤖";

  // Bubble
  const bubble = document.createElement("div");
  bubble.className = "message-bubble";

  if (!isUser && typeof marked !== "undefined") {
    bubble.innerHTML = marked.parse(text);
  } else {
    bubble.textContent = text;
  }

  row.appendChild(avatar);
  row.appendChild(bubble);
  chat.messages.appendChild(row);
  
  smoothScrollToBottom();
}

// Typing indicator (AI is thinking)
function showTyping() {
  const row = document.createElement("div");
  row.id = "typing";
  row.className = "chat-row bot-row new-sender";
  
  const avatar = document.createElement("div");
  avatar.className = "chat-avatar";
  avatar.innerHTML = "🤖";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.innerHTML = `
    <div class="typing-indicator">
      <span></span><span></span><span></span>
    </div>
  `;

  row.appendChild(avatar);
  row.appendChild(bubble);
  chat.messages.appendChild(row);
  smoothScrollToBottom();
  
  return row;
}

// Gửi tin nhắn
async function sendMessage() {
  const text = chat.input.value.trim();
  if (!text) return;

  addChatMessage(text, true);
  chat.input.value = "";

  const typingIndicator = showTyping();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });

    const data = await res.json();
    typingIndicator.remove();

    if (res.ok) {
      addChatMessage(
        data.reply || "Tôi chưa hiểu rõ câu hỏi này. Bạn thử hỏi chi tiết hơn nhé!"
      );
    } else {
      addChatMessage("⚠️ Server đang bận, vui lòng thử lại sau.");
    }
  } catch (err) {
    if (typingIndicator) typingIndicator.remove();
    addChatMessage("❌ Không thể kết nối với AI. Vui lòng kiểm tra mạng.");
  }
}

// Events
if (chat.send) chat.send.addEventListener("click", sendMessage);
if (chat.input) {
  chat.input.addEventListener("keypress", (e) => {
    if (e.key === "Enter") sendMessage();
  });
}

// Welcome message
document.addEventListener("DOMContentLoaded", () => {
  setTimeout(() => {
    if (chat.messages && chat.messages.children.length === 0) {
      addChatMessage(
        "👋 Chào bạn! Mình là **AI tư vấn tuyển sinh 2026**.\n\nBạn có thể hỏi về ngành học, trường đại học hoặc điểm chuẩn bất kỳ năm nào.\n\n💡 Để tính xác suất trúng tuyển chính xác hơn, hãy dùng **form tư vấn** bên trên nhé!"
      );
    }
  }, 800);
});
