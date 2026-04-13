// ====================== CHATBOT.JS - ĐÃ TỐI ƯU ======================

const chat = {
  toggle: document.getElementById("chat-toggle"),
  window: document.getElementById("chat-window"),
  messages: document.getElementById("chat-messages"),
  input: document.getElementById("chat-input"),
  send: document.getElementById("chat-send"),
  close: document.getElementById("chat-close"),
};

// Toggle chatbot
if (chat.toggle && chat.window) {
  chat.toggle.addEventListener("click", () => {
    chat.window.classList.toggle("hidden");
    if (!chat.window.classList.contains("hidden")) {
      chat.input.focus();
    }
  });
}

if (chat.close) {
  chat.close.addEventListener("click", () => {
    chat.window.classList.add("hidden");
  });
}

// Thêm tin nhắn
function addChatMessage(text, isUser = false) {
  const div = document.createElement("div");
  div.className = `chat-message ${isUser ? "user-message" : "bot-message"}`;

  div.innerHTML = `<div class="message-bubble">${text}</div>`;
  chat.messages.appendChild(div);
  chat.messages.scrollTop = chat.messages.scrollHeight;
}

// Typing indicator
function showTyping() {
  const typing = document.createElement("div");
  typing.id = "typing";
  typing.className = "chat-message bot-message";
  typing.innerHTML = `<div class="message-bubble typing"><span></span><span></span><span></span></div>`;
  chat.messages.appendChild(typing);
  chat.messages.scrollTop = chat.messages.scrollHeight;
  return typing;
}

// Gửi tin nhắn
async function sendMessage() {
  const text = chat.input.value.trim();
  if (!text) return;

  addChatMessage(text, true);
  chat.input.value = "";

  const typingIndicator = showTyping();

  try {
    const res = await fetch("/api/tu-van", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        diem_thi: 0,
        khu_vuc: "KV3",
        ma_nganh: "",
        to_hop: "",
        query: text,
      }),
    });

    const data = await res.json();
    typingIndicator.remove();

    if (res.ok) {
      addChatMessage(
        data.ket_qua ||
          "Tôi chưa hiểu rõ câu hỏi này. Bạn thử hỏi chi tiết hơn nhé!"
      );
    } else {
      addChatMessage("⚠️ Server đang bận, vui lòng thử lại sau.");
    }
  } catch (err) {
    typingIndicator.remove();
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

// Lời chào khi mở chatbot lần đầu
document.addEventListener("DOMContentLoaded", () => {
  setTimeout(() => {
    if (chat.messages && chat.messages.children.length === 0) {
      addChatMessage(
        "👋 Chào bạn! Mình là AI tư vấn tuyển sinh 2026.<br>Bạn muốn hỏi về ngành nào, trường nào hay điểm chuẩn?"
      );
    }
  }, 600);
});
