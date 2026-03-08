let allMajors = []; // Lưu trữ biến toàn cục

// Load danh sách ngành khi trang vừa hiển thị
document.addEventListener("DOMContentLoaded", async () => {
    try {
        const response = await fetch("/api/majors");
        const data = await response.json();
        allMajors = data.majors;
        renderDropdown(allMajors);
    } catch (error) {
        console.error("Lỗi khi tải danh mục ngành:", error);
    }
});

const searchInput = document.getElementById("ma_nganh");
const dropdownList = document.getElementById("majors_dropdown");

function renderDropdown(majorsList) {
    dropdownList.innerHTML = "";
    if (majorsList.length === 0) {
        dropdownList.innerHTML = "<div class='dropdown-item' style='color: #888;'>Không tìm thấy ngành phù hợp</div>";
        return;
    }
    
    majorsList.forEach(major => {
        const item = document.createElement("div");
        item.className = "dropdown-item";
        item.innerHTML = `<strong>${major.ma_nganh}</strong> - ${major.ten_nganh}`;
        
        // Khi click chọn 1 ngành
        item.addEventListener("click", () => {
            searchInput.value = major.ma_nganh; // Chỉ lấy mã ngành
            dropdownList.style.display = "none";
        });
        
        dropdownList.appendChild(item);
    });
}

// Xử lý sự kiện gõ tìm kiếm
searchInput.addEventListener("input", (e) => {
    const keyword = e.target.value.toLowerCase().trim();
    if (keyword === "") {
        renderDropdown(allMajors);
    } else {
        const filtered = allMajors.filter(m => 
            m.ma_nganh.toLowerCase().includes(keyword) || 
            m.ten_nganh.toLowerCase().includes(keyword)
        );
        renderDropdown(filtered);
    }
    dropdownList.style.display = "block";
});

// Hiển thị dropdown khi click vào ô input
searchInput.addEventListener("focus", () => {
    dropdownList.style.display = "block";
});

// Ẩn dropdown khi click ra ngoài
document.addEventListener("click", (e) => {
    if (!searchInput.contains(e.target) && !dropdownList.contains(e.target)) {
        dropdownList.style.display = "none";
    }
});

function toggleCalculator() {
    const calc = document.getElementById("calculator-section");
    calc.style.display = calc.style.display === "none" ? "block" : "none";
}

function calculateBestCombo() {
    const s = {
        toan: parseFloat(document.getElementById("s_toan").value) || -1,
        van: parseFloat(document.getElementById("s_van").value) || -1,
        anh: parseFloat(document.getElementById("s_anh").value) || -1,
        ly: parseFloat(document.getElementById("s_ly").value) || -1,
        hoa: parseFloat(document.getElementById("s_hoa").value) || -1,
        sinh: parseFloat(document.getElementById("s_sinh").value) || -1,
        su: parseFloat(document.getElementById("s_su").value) || -1,
        dia: parseFloat(document.getElementById("s_dia").value) || -1,
        gdcd: parseFloat(document.getElementById("s_gdcd").value) || -1
    };
    
    const combos = [
        { name: "A00", label: "A00 (Toán, Vật lý, Hóa học)", total: s.toan + s.ly + s.hoa, valid: s.toan>=0 && s.ly>=0 && s.hoa>=0 },
        { name: "A01", label: "A01 (Toán, Vật lý, Tiếng Anh)", total: s.toan + s.ly + s.anh, valid: s.toan>=0 && s.ly>=0 && s.anh>=0 },
        { name: "D01", label: "D01 (Toán, Ngữ văn, Tiếng Anh)", total: s.toan + s.van + s.anh, valid: s.toan>=0 && s.van>=0 && s.anh>=0 },
        { name: "D07", label: "D07 (Toán, Hóa học, Tiếng Anh)", total: s.toan + s.hoa + s.anh, valid: s.toan>=0 && s.hoa>=0 && s.anh>=0 },
        { name: "B00", label: "B00 (Toán, Hóa học, Sinh học)", total: s.toan + s.hoa + s.sinh, valid: s.toan>=0 && s.hoa>=0 && s.sinh>=0 },
        { name: "C00", label: "C00 (Ngữ văn, Lịch sử, Địa lý)", total: s.van + s.su + s.dia, valid: s.van>=0 && s.su>=0 && s.dia>=0 },
        { name: "A02", label: "A02 (Toán, Vật lý, Sinh học)", total: s.toan + s.ly + s.sinh, valid: s.toan>=0 && s.ly>=0 && s.sinh>=0 }
    ];
    
    let best = null;
    for(let c of combos) {
        if(c.valid) {
            if(!best || c.total > best.total) best = c;
        }
    }
    
    if(!best) {
        alert("Thất bại: Vui lòng nhập đủ điểm cho ít nhất 1 tổ hợp thi hợp lệ (Ví dụ: phải nhập đủ cả Toán, Lý, Hóa để hệ thống tính khối A00)!");
        return;
    }
    
    // Check if combo is in the select dropdown, add it if missing
    const select = document.getElementById("to_hop");
    let optionExists = false;
    for(let i=0; i<select.options.length; i++) {
        if(select.options[i].value === best.name) optionExists = true;
    }
    
    if(!optionExists) {
        const opt = document.createElement("option");
        opt.value = best.name;
        opt.text = best.label;
        select.appendChild(opt);
    }
    
    // Set values automatically
    select.value = best.name;
    document.getElementById("diem_thi").value = best.total.toFixed(2);
    
    alert(`🎉 Thành công! Tổ hợp cao điểm nhất của bạn là ${best.name} với ${best.total.toFixed(2)} điểm. Hệ thống đã tự động nhập vào Form!`);
    toggleCalculator();
}

async function hoiDap() {
    const diem_thi = document.getElementById("diem_thi").value;
    const khu_vuc = document.getElementById("khu_vuc").value;
    const ma_nganh = document.getElementById("ma_nganh").value;
    const to_hop = document.getElementById("to_hop").value;
    const query = document.getElementById("query").value;
    
    if (!diem_thi || !ma_nganh || !to_hop) {
        alert("Vui lòng nhập điểm thi, chọn tổ hợp và ngành ưu tiên!");
        return;
    }
    
    const resultCard = document.getElementById("result-card");
    const loading = document.getElementById("loading");
    const resultContent = document.getElementById("result-content");
    
    resultCard.style.display = "block";
    loading.style.display = "block";
    resultContent.style.display = "none";
    
    try {
        const response = await fetch("/api/tu-van", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                diem_thi: parseFloat(diem_thi),
                khu_vuc: khu_vuc,
                ma_nganh: ma_nganh,
                to_hop: to_hop,
                query: query
            })
        });
        
        const data = await response.json();
        
        loading.style.display = "none";
        resultContent.style.display = "block";
        resultContent.innerText = data.ket_qua;
    } catch (error) {
        loading.style.display = "none";
        resultContent.style.display = "block";
        resultContent.innerText = "Có lỗi xảy ra trong quá trình kết nối với server: " + error.message;
    }
}
