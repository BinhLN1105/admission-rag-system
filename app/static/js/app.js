// ====================== APP.JS ======================

const elements = {
  diemThi: document.getElementById("diem_thi"),
  khuVuc: document.getElementById("khu_vuc"),
  toHop: document.getElementById("to_hop"),
  maNganh: document.getElementById("ma_nganh"),
  query: document.getElementById("query"),
  majorsDropdown: document.getElementById("majors_dropdown"),
  resultCard: document.getElementById("result-card"),
  loading: document.getElementById("loading"),
  resultOriginal: document.getElementById("result-original"),
  resultLLM: document.getElementById("result-llm"),
  llmContent: document.getElementById("llm-content"),
  llmMiniLoader: document.getElementById("llm-mini-loader"),
  viewToggle: document.getElementById("view-toggle"),
};

let allMajors = [];

// Load danh sách ngành
async function loadMajors() {
  try {
    const res = await fetch(`/api/majors?v=${Date.now()}`);
    if (!res.ok) throw new Error("Failed to load majors");

    const data = await res.json();
    allMajors = data.majors || [];
    renderDropdown(allMajors);
  } catch (err) {
    console.error("Lỗi load majors:", err);
    elements.majorsDropdown.innerHTML = `<div class="dropdown-item error">Không tải được danh sách ngành</div>`;
  }
}

function renderDropdown(majors) {
  elements.majorsDropdown.innerHTML = "";

  if (!majors.length) {
    elements.majorsDropdown.innerHTML = `<div class="dropdown-item">Không tìm thấy ngành phù hợp</div>`;
    return;
  }

  const fragment = document.createDocumentFragment();
  majors.forEach((major) => {
    const item = document.createElement("div");
    item.className = "dropdown-item";
    item.innerHTML = `<strong>${major.ma_nganh}</strong> — ${major.ten_nganh}`;
    item.onclick = () => {
      elements.maNganh.value = major.ma_nganh;
      elements.majorsDropdown.style.display = "none";
    };
    fragment.appendChild(item);
  });
  elements.majorsDropdown.appendChild(fragment);
}

// Search với debounce
const searchHandler = utils.debounce((e) => {
  const keyword = e.target.value.toLowerCase().trim();

  const filtered = keyword
    ? allMajors.filter(
        (m) =>
          m.ma_nganh.toLowerCase().includes(keyword) ||
          m.ten_nganh.toLowerCase().includes(keyword)
      )
    : allMajors;

  renderDropdown(filtered);
  elements.majorsDropdown.style.display = "block";
}, 160);

elements.maNganh.addEventListener("input", searchHandler);
elements.maNganh.addEventListener("focus", () => {
  if (allMajors.length) elements.majorsDropdown.style.display = "block";
});

// Click outside to close dropdown
document.addEventListener("click", (e) => {
  if (
    !elements.maNganh.contains(e.target) &&
    !elements.majorsDropdown.contains(e.target)
  ) {
    elements.majorsDropdown.style.display = "none";
  }
});

// Main submit function
window.hoiDap = async function () {
  const payload = {
    diem_thi: parseFloat(elements.diemThi.value) || 0,
    khu_vuc: elements.khuVuc.value,
    ma_nganh: elements.maNganh.value.trim(),
    to_hop: elements.toHop.value,
    query: elements.query.value.trim(),
  };

  if (!payload.diem_thi || !payload.ma_nganh || !payload.to_hop) {
    utils.showToast("Vui lòng nhập đầy đủ thông tin bắt buộc!", "error");
    return;
  }

  // Show loading
  elements.resultCard.style.display = "block";
  elements.loading.style.display = "block";
  elements.resultOriginal.classList.remove('active');
  elements.resultLLM.classList.remove('active');
  elements.viewToggle.style.display = "none";

  try {
    // 1. GỌI API TƯ VẤN NHANH (BẢN GỐC + FACTS)
    const res = await fetch("/api/tu-van", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();
    elements.loading.style.display = "none";
    
    // Render kết quả gốc ngay lập tức
    elements.resultOriginal.innerHTML = marked.parse(data.ket_qua || "Không có kết quả.");
    
    // Setup Tab AI ở trạng thái chờ
    elements.viewToggle.style.display = "flex";
    elements.llmContent.innerHTML = "";
    elements.llmMiniLoader.style.display = "inline-flex";
    
    // Mặc định hiển thị tab AI (đang loading)
    switchView('llm');

    utils.showToast("Đã có kết quả tư vấn nhanh!", "success");

    // 2. GỌI API LLM (PHÂN TÍCH CHUYÊN SÂU) - CHẠY NGẦM
    if (data.facts) {
      try {
        const llmRes = await fetch("/api/tu-van-llm", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(data.facts),
        });
        const llmData = await llmRes.json();
        
        elements.llmMiniLoader.style.display = "none";
        if (llmData.ket_qua_llm) {
          elements.llmContent.innerHTML = marked.parse(llmData.ket_qua_llm);
        } else {
          elements.llmContent.innerHTML = "<p>⚠️ Không thể khởi tạo phân tích AI lúc này. Bạn vui lòng xem bản gốc nhé.</p>";
        }
      } catch (llmErr) {
        console.error("LLM API Error:", llmErr);
        elements.llmMiniLoader.style.display = "none";
        elements.llmContent.innerHTML = "<p>❌ Lỗi kết nối dịch vụ AI.</p>";
      }
    } else {
      elements.llmMiniLoader.style.display = "none";
      elements.llmContent.innerHTML = "<p>ℹ️ Không có đủ dữ liệu để phân tích AI chuyên sâu.</p>";
    }

  } catch (err) {
    console.error("Main API Error:", err);
    elements.loading.style.display = "none";
    elements.resultOriginal.innerHTML = `<p style="color:#ef4444;">❌ Lỗi kết nối server. Vui lòng thử lại.</p>`;
    elements.viewToggle.style.display = "none";
    switchView('original');
  }
};

// Toggle giữa các view
window.switchView = function(view) {
  const views = ['original', 'llm'];
  views.forEach(v => {
    const el = document.getElementById(`result-${v}`);
    const btn = document.querySelector(`.btn-toggle[onclick*="'${v}'"]`);
    if (v === view) {
      el.classList.add('active');
      if (btn) btn.classList.add('active');
    } else {
      el.classList.remove('active');
      if (btn) btn.classList.remove('active');
    }
  });
};

// Initialize
document.addEventListener("DOMContentLoaded", () => {
  loadMajors();
});
