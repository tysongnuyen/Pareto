# Pareto MAPPO với PRB chung

Thư mục `Pareto Model` là một bộ mã nguồn độc lập: có sẵn bản sao simulator,
networks, PPO–GAE và training loop; không import từ thư mục cha và không dùng
symlink. Có thể copy nguyên thư mục lên server. Các file dùng chung ở thư mục
Astra feedback và bản `Pareto RL/` cũ được giữ nguyên.

| File | Vai trò |
|---|---|
| `train_pareto_mappo.py`, `run_pareto_mappo.sh` | Khởi chạy bản Pareto |
| `pareto_mappo.py` | Env Pareto và model đã chốt |
| `local_mappo_baseline.py` | PPO–GAE allocation |
| `strict_local_independent_mappo.py` | Networks, placement PPO và inner loop dùng lại |
| `env.py`, `ver2.py` | Mô hình vật lý và Gym environment |
| `learning_credit.py` | Các phép tính policy/entropy dùng chung |
| `train_strict_local_mappo.py` | Lịch train, log, validation, checkpoint/resume |
| `kmeans_baseline.py` | Helper được runner và self-check import; baseline này mặc định tắt |
| `test_pareto_mappo.py` | Kiểm tra mask, PPO, critic, checkpoint và số UE |

Chạy **`train_pareto_mappo.py`**, không chạy trực tiếp các runner phụ thuộc.
Các bản sao dùng chung là snapshot; sửa model ở thư mục cha không tự thay đổi
bản trong thư mục này.

## Thiết kế

- 5 placement actor độc lập, attention và quan sát toàn cục; placement critic
  dùng PopArt như model chính.
- 5 allocation actor độc lập: attention trên tập UE local (9 đặc trưng/UE,
  vẫn có relative x/y), MLP local 11 chiều và residual.
- Vector 11 chiều: PRB chung còn lại / 135, power của UAV / 2 W, QoS UE
  hiện tại, 3 link feature và one-hot UAV ID. Không thêm hình học UAV,
  relative x/y của UE hiện tại, giới hạn cố định hay broadcast vào vector này.
- `coordination_mode=none`; PRB là một quỹ chung 135, power riêng 2 W/UAV.
- Pareto chỉ giữ cấp phát trong vùng phủ, đủ QoS, đủ tài nguyên và không bị
  chi phối về PRB/power. Reject luôn hợp lệ. Lưới gốc: PRB 1–10, power
  0.1–1.0 W. Mỗi PRB hợp lệ có một power duy nhất.
- Actor học participate và PRB. Không có power head có tham số: API categorical
  power được giữ để tái sử dụng collector, với đúng một lựa chọn xác suất 1.
- Allocation critic giống kiến trúc global attention cũ: UE, UAV, link và phase,
  không candidate set, không bảng chi phí, không Pareto mask làm đầu vào mạng,
  không PopArt. Simulator không tính bảng chi phí critic trong bản này.
- Allocation dùng sampled-action PPO + GAE (`gamma=1`, `lambda=0.95`), advantage
  chung cho đội; critic học mọi transition, actor bỏ các quyết định chỉ có reject.
  Không gọi simulator preview hay tính Q counterfactual. Các catalog xác suất
  còn lại chỉ để tính entropy/KL.
- Giữ chuẩn hóa return theo số UE train, chuẩn hóa advantage và entropy theo
  nhánh có trọng số detached của learner hiện tại; đây không phải allocation PopArt.
- Reward giữ nguyên: `1 - 0.2*P_winner/2 - n_winner/135`; không winner thì 0.
  Bộ phân xử bid, ràng buộc vật lý và thứ tự UE không thay đổi.

## Lịch mặc định

- 710 global = 10 warm-up allocation + 700 joint; không placement-only/freeze.
- Mỗi global lấy 2 map, mỗi map dùng đủ 5 thứ tự UAV vòng: 10 placement sample.
  Có 1.420 lượt map train; 5 thứ tự không bảo đảm cho 5 bộ vị trí khác nhau.
- Inner loop giữ cố định 10 placement; mỗi inner update thu 10 allocation
  episode. Với 50 UE là 500 transition/update, minibatch 500, 5 PPO epoch.
- Tối đa 20 inner update/global, patience 10 lần đánh giá deterministic
  không cải thiện; giữ model cuối, không rollback best.
- Placement tích lũy 10 global = 100 sample rồi cập nhật PPO, 5 epoch,
  minibatch 10. Target placement là kết quả zero-shot trước inner adaptation,
  giữ nguyên runner hiện tại. Khi batch đã đủ, placement cập nhật sau đánh giá
  zero-shot và trước inner loop allocation của global đó; các vị trí đã lấy
  mẫu vẫn giữ nguyên cho inner loop.
- LR actor 2e-4, critic 3e-4; entropy allocation 0.003 cố định; train seed 2026.
- Lưu best theo validation để test cuối, latest resume tại ranh giới batch
  placement hoàn chỉnh (mặc định mỗi 10 global). `latest_inner_model.pt` chỉ
  dùng đánh giá, không phải checkpoint resume đầy đủ.
- Dừng PPO theo KL từng actor (`target_kl=0.02`) khác với patience inner.

## Chạy

```bash
cd "/home/honganh/Documents/Astra feedback/Pareto Model"
/home/honganh/miniconda3/envs/myrl/bin/python -B -u train_pareto_mappo.py \
  --device cpu \
  --output-dir training_outputs_pareto_mappo_seed2026
```

Hoặc trên server, copy nguyên thư mục `Pareto Model`, vào thư mục đó và
activate môi trường có NumPy, PyTorch, Gymnasium, Matplotlib:

```bash
bash run_pareto_mappo.sh --device cuda --output-dir training_outputs_pareto_mappo_seed2026
```

`PARETO_PYTHON` có thể chọn interpreter. `--print-config` xem cấu hình mà không
train. Không cần sửa đường dẫn trong mã nguồn.

Resume dùng đúng checkpoint của bản này, đúng source/runtime và thư mục run:

```bash
python -B -u train_pareto_mappo.py \
  --resume training_outputs_pareto_mappo_seed2026/checkpoints/latest_resume.pt
```

Checkpoint coverage/COMA/Pareto cũ không tương thích với kiến trúc mới; bắt đầu
run mới. Không ghi đè các output đã có.

Khi đánh giá checkpoint train 50 UE trên số UE khác, khởi tạo learner với
`reward_scale` lấy từ checkpoint (50), không đổi thang critic theo số UE test.
Attention không cần thay kích thước trọng số khi thay số UE.

Kiểm tra nhỏ, không chạy experiment dài:

```bash
python -B train_pareto_mappo.py --self-check
```
