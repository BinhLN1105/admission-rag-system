// ====================== UTILS.JS ======================

// Debounce function
function debounce(func, delay = 180) {
  let timer;
  return function (...args) {
    clearTimeout(timer);
    timer = setTimeout(() => func.apply(this, args), delay);
  };
}

// Format number kiểu Việt Nam
function formatNumber(num) {
  return num.toLocaleString("vi-VN");
}

// Toast notification hiện đại
function showToast(message, type = "success") {
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.textContent = message;

  document.body.appendChild(toast);

  // Auto remove
  setTimeout(() => {
    toast.style.opacity = "0";
    setTimeout(() => toast.remove(), 400);
  }, 2800);
}

// Export
window.utils = {
  debounce,
  formatNumber,
  showToast,
};
