# Báo cáo Lab Day 27 – Track 03

- **Họ tên:** Hà Trung Dũng
- **MSV:** 2A202602948
- **Bài lab:** Human-in-the-Loop PR Review Agent

## 1. Mục tiêu

Xây dựng agent review Pull Request bằng LangGraph: đọc metadata/diff từ GitHub,
phân tích có cấu trúc bằng LLM, định tuyến theo confidence, dừng để nhận quyết
định của con người khi cần, lưu checkpoint bền vững và ghi audit trail có thể
truy vấn/replay. Streamlit cung cấp giao diện cho toàn bộ workflow.

## 2. Công nghệ

Python 3.12, uv, LangGraph, LangChain/OpenRouter, Pydantic, httpx, aiosqlite,
AsyncSqliteSaver, Rich, Streamlit và pytest.

## 3. Kiến trúc HITL Agent

Luồng chính:

`fetch_pr -> analyze -> route`

- `confidence >= 0.73`: `auto_approve` và đăng comment trực tiếp.
- `0.58 <= confidence < 0.73`: `human_approval` (`interrupt`) rồi
  `commit` nếu approve/edit; reject không đăng.
- `confidence < 0.58`: `escalate` (`interrupt`) -> `synthesize`; trên
  Streamlit có thêm `final_confirmation` trước `commit`.

Mỗi session dùng một `thread_id`. Checkpoint LangGraph và bảng
`audit_events` cùng nằm trong `hitl_audit.db` nhưng phục vụ hai mục đích khác
nhau: resume workflow và replay quyết định.

## 4. Kết quả từng Exercise

### Exercise 1 – Confidence Routing

Đã hoàn thiện fetch PR, structured output `PRAnalysis`, prompt review chi tiết,
routing đúng ngưỡng 0.58/0.73 và graph conditional edges.

### Exercise 2 – Human-in-the-Loop

Đã dùng `interrupt()`, `MemorySaver`, `Command(resume=...)` đúng `thread_id`.
Ba lựa chọn approve/reject/edit hoạt động; edit đăng đúng Markdown đã chỉnh sửa,
reject không gọi GitHub.

### Exercise 3 – Escalation

Nhánh confidence thấp yêu cầu 2–4 câu hỏi gắn với rủi ro/diff. Câu trả lời được
lưu vào state và đưa cùng diff, initial analysis vào lần synthesize. Confidence
sau synthesize không bị ép tăng.

### Exercise 4 – Structured Audit Trail

Đã dùng `AsyncSqliteSaver`, audit đủ các node, ghi quyết định/feedback của con
người, đo thời gian bằng `time.monotonic()`, suy ra risk bằng
`risk_level_for()`. `event_key` và unique index tránh bản ghi lặp khi node chạy
lại sau interrupt. CLI `--thread` đọc checkpoint cũ thay vì chạy lại từ đầu.
`worst_risk` được tính theo thứ tự high > med > low, không so sánh chuỗi.

### Exercise 5 – Streamlit UI

Đã hoàn thiện form PR, trạng thái chạy/lỗi, ba giao diện auto/human/escalation,
edit review, final confirmation cho refined review, recent sessions và mở lại
checkpoint cũ. UI chỉ báo đăng thành công khi GitHub trả thành công; dry-run và
lỗi commit có trạng thái riêng.

## 5. An toàn và chống trùng

- `.env`, database, virtualenv và cache được loại khỏi Git.
- Không ghi secrets vào source/log/report.
- Comment có marker theo `thread_id`; trước khi POST, client tìm marker để tránh
  đăng lại trong kịch bản retry đã hỗ trợ.
- `HITL_DRY_RUN=true` bảo đảm không POST comment thật.
- Output LLM mặc định giới hạn 2.048 tokens để tránh dự trữ chi phí không cần thiết.

## 6. Cách chạy

```powershell
uv sync
uv run python exercises/exercise_1_confidence.py --pr https://github.com/VinUni-AI20k/PR-Demo/pull/1
uv run python exercises/exercise_2_hitl.py --pr https://github.com/VinUni-AI20k/PR-Demo/pull/1
uv run python exercises/exercise_3_escalation.py --pr https://github.com/VinUni-AI20k/PR-Demo/pull/2
uv run python exercises/exercise_4_audit.py --pr https://github.com/VinUni-AI20k/PR-Demo/pull/1
uv run streamlit run app.py
```

Resume một session durable:

```powershell
uv run python exercises/exercise_4_audit.py --thread <thread_id>
```

Audit:

```powershell
uv run python -m audit.replay --list
uv run python -m audit.replay --thread <thread_id>
```

## 7. Kết quả kiểm thử thực tế

- `uv sync`: thành công.
- `uv run python -m compileall exercises common audit app.py`: thành công.
- `uv run pytest -q`: **21 passed, 0 failed, 0 skipped**.
- Audit schema/write/list/replay và durable resume qua hai kết nối SQLite độc lập:
  đã kiểm thử bằng pytest.
- Streamlit AppTest: không có exception.
- Streamlit server thật: khởi động thành công; `/_stcore/health` trả `ok`.
- `uv run python -m audit.replay --list`: chạy thành công.
- `git diff --check`: không phát hiện whitespace error.

Smoke test ban đầu với GitHub + OpenRouter ngày 15/10/2026, không đăng comment:

- PR #1: fetch thành công 3 files; LLM trả confidence 55%, 4 comments; route
  `escalate`.
- PR #2: fetch thành công 4 files; LLM trả confidence 55%, 4 comments; route
  `escalate`.

Hai score bằng nhau do prompt ban đầu chưa định nghĩa rõ confidence là độ an toàn
để agent tự hoàn tất review. Prompt đồng thời nhắc mốc 58% ngay cạnh yêu cầu tạo
câu hỏi nhưng không có rubric tính điểm, khiến model nhỏ dễ neo vào 0.55. Schema
cũ còn có thể được hiểu là độ chắc chắn rằng finding đúng; cách hiểu đó không phù
hợp khi dùng confidence làm proxy cho risk routing.

Đã bổ sung rubric dùng chung cho Exercise 1–4:

- 0.73–1.00: thay đổi nhỏ, cục bộ, ít rủi ro và không còn vấn đề quan trọng.
- 0.58–0.72: rủi ro vừa như additive schema migration, validation hoặc thiếu test.
- 0.00–0.57: auth/security, weak hashing/token, plaintext secret, SQL injection,
  non-TLS sync, hard-coded identity hoặc nhiều rủi ro lớn tương tác.

Rubric định nghĩa score là mức an toàn để xử lý tự động, không phải độ chắc rằng
một bug tồn tại; model phải giải thích các yếu tố tạo score và không được chọn số
chỉ để kích hoạt nhánh. `PRAnalysis.confidence` cũng được cập nhật cùng semantics.
Không có URL, PR number hoặc confidence riêng cho PR demo trong prompt/code.

Sau thay đổi, đã thử live verification đúng một lần cho mỗi PR. GitHub fetch thành
công nhưng OpenRouter trả HTTP 402 trước khi sinh output vì số dư chỉ đủ khoảng 764
output tokens, thấp hơn giới hạn 2.048. Không retry thêm, không hạ token xuống mức
dễ làm structured review bị cắt cụt, và không có comment nào được đăng. Vì vậy
calibration mới **chưa được xác minh live**. Routing biên vẫn được xác nhận offline
tại 0.57, 0.58, 0.72, 0.73 và 0.90.

## 8. Các điểm sửa so với skeleton

- Hoàn tất mọi TODO/`NotImplementedError` thuộc Exercise 1–5.
- Sửa edit bị bỏ qua, resume sai `thread_id`, audit lặp sau interrupt, so sánh
  `worst_risk` sai, trạng thái đăng GitHub giả thành công và nguy cơ duplicate.
- Thêm final confirmation thật cho refined escalation review trên UI.
- Sửa lỗi Unicode terminal trên Windows và giới hạn output tokens của LLM.
- Bổ sung 21 test offline dùng mock GitHub/LLM, gồm kiểm tra mọi Exercise dùng
  cùng calibration prompt và prompt không chứa GitHub URL; không tạo comment thật.

## 9. Hạn chế

- Confidence/routing live phụ thuộc model và không đảm bảo giống con số minh họa
  trong README.
- Calibration prompt mới chưa thể live-verify do OpenRouter hết hạn mức phù hợp;
  cần thêm credit hoặc dùng key/model còn quota rồi chạy lại sau.
- Luồng đăng comment thật không được thực hiện để tránh tạo dữ liệu trên PR demo;
  hành vi POST, lỗi, retry/idempotency và các nhánh workflow đã được kiểm thử bằng mock.
- GitHub vẫn cần PAT hợp lệ và quyền phù hợp khi người dùng chủ động chạy chế độ thật.
