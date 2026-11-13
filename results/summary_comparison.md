| Phương pháp | Sạch (10k test) | Sạch (1k) | FGSM 8/255 | PGD-10 8/255 | PGD-20 8/255 | Tấn công thích nghi 8/255 | Huấn luyện (giây/epoch) | Suy luận (ms/ảnh) |
|---|---|---|---|---|---|---|---|---|
| Baseline | 93.8 | 94.0 | 21.7 | 0.0 | 0.0 | 0.0 | – | 3.06 |
| Fast-AT | 84.2 | 84.3 | 55.3 | 48.2 | 46.7 | 46.7 | 63 | 3.06 |
| PGD-AT | 81.0 | 81.3 | 55.2 | 51.0 | 49.5 | 49.5 | 219 | 3.06 |
| Baseline + JPEG q=90 | – | 90.4 | 27.6 | 1.3 | – | 0.3 | 0 | 3.16 |
| Baseline + JPEG q=75 | – | 83.5 | 39.8 | 43.2 | – | 3.3 | 0 | 3.17 |
| Baseline + JPEG q=50 | – | 76.6 | 51.6 | 62.3 | – | 8.6 | 0 | 3.13 |
| Baseline + 5-bit | – | 93.4 | 22.5 | 0.0 | – | 0.0 | 0 | 3.09 |
| Baseline + 4-bit | – | 91.4 | 22.3 | 0.0 | – | 0.0 | 0 | 3.10 |
| Baseline + 3-bit | – | 84.2 | 26.9 | 3.8 | – | 0.2 | 0 | 3.04 |
| Baseline + Blur σ=0.5 | – | 91.1 | 24.3 | 0.1 | – | 0.2 | 0 | 3.10 |
| Baseline + Blur σ=1.0 | – | 31.6 | 26.3 | 24.9 | – | 1.8 | 0 | 3.14 |

Ghi chú:
- Accuracy dưới tấn công đo trên 1.000 ảnh test cố định (seed 42), ε theo chuẩn L∞, white-box, untargeted.
- Baseline/AT: FGSM, PGD-10 và PGD-20 tấn công trực tiếp mô hình, nên PGD-20 cũng là tấn công thích nghi.
- Tiền xử lý: FGSM/PGD-10 sinh trên baseline rồi mới qua bước tiền xử lý (kẻ tấn công không biết phòng thủ); cột 'thích nghi' là PGD-10 + BPDA tấn công trực tiếp f(g(x)).
- Số liệu baseline lấy từ cùng lượt chạy đánh giá với AT (adv_training_*_metrics.json) để so sánh cùng điều kiện; lượt đo riêng của phần tấn công (attack_metrics.json, CPU) lệch ≤ 0,1 điểm.
- Thời gian huấn luyện đo trên GPU Colab; baseline không ghi thời gian mỗi epoch. Tiền xử lý không cần huấn luyện lại.
- Suy luận: CPU (AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD), 4 luồng, batch 128, 1000 ảnh, trung vị 7 lần đo luân phiên; Fast-AT/PGD-AT cùng kiến trúc với baseline nên cùng chi phí suy luận.
