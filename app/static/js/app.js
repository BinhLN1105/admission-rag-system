async function hoiDap() {
    const diem_thi = document.getElementById("diem_thi").value;
    const khu_vuc = document.getElementById("khu_vuc").value;
    const ma_nganh = document.getElementById("ma_nganh").value;
    const query = document.getElementById("query").value;
    
    if (!diem_thi || !ma_nganh || !query) {
        alert("Vui lòng nhập đầy đủ thông tin!");
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
