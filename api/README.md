# Project 2 API: Malaria Cell Classifier + Case Note Analyzer

Week 8 turns the Project 2 system into a tested web service. It serves my Week 6 image model (ResNet-18, classes `Parasitized` / `Uninfected`) and my Week 7 text evidence rules. It classifies a blood-cell image, masks private details in a clinical note, flags urgent words, and returns a `REFER` or `ROUTINE` decision for a human to review. All code is in the [`api`](api) folder.

> Decision support only. A qualified human decides.

### Files

| File | Purpose |
|---|---|
| [`api/app.py`](api/app.py) | FastAPI service with `/health`, `/predict`, `/analyze` |
| [`api/test_api.py`](api/test_api.py) | 8 automated tests |
| [`api/bench.py`](api/bench.py) | Latency benchmark against a running server |
| [`api/sample.jpeg`](api/sample.jpeg) | Test image used by the tests and the benchmark |
| [`api/README.md`](api/README.md) | Same documentation, kept next to the code |

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Shows the model is loaded, the classes, and the positive class |
| POST | `/predict` | Upload one JPEG/PNG image, get label, probabilities and latency |
| POST | `/analyze` | Upload an image plus a text note, get a full case record |

Settings (environment variables): `MODEL_PATH` (default `week6_model.pt`), `POSITIVE` (default `Parasitized`, the class that must not be missed), `THRESHOLD` (default `0.5`), `API_KEY` (optional).

### Model details

- Checkpoint: `week6_model.pt` (about 43 MB, saved in Week 6). It is **not committed** to this repository because of its size; place it in the `api` folder before running.
- Backbone: ResNet-18 (transfer learning), 224 x 224 RGB input, ImageNet normalisation. The service rebuilds the network and the preprocessing from the settings stored in the checkpoint, so the API and training use the same transform.
- Week 6 test-set result (3,335 images): macro-F1 0.969, accuracy 0.970. Parasitized: precision (PPV) 0.964, recall (sensitivity) 0.961. Uninfected: precision 0.974, recall (specificity) 0.976. Confusion matrix (rows = true, columns = predicted, order Parasitized, Uninfected): [[1288, 52], [48, 1947]], so 52 infected cells were missed and 48 healthy cells were flagged. The test set is about 40% Parasitized, so the PPV would be lower in a setting where infection is rarer.
- Urgent words checked in notes: fever, chills, vomiting, breath, confusion, jaundice, seizure, weakness.

### How to run

```
cd api
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn python-multipart pillow pytest httpx numpy
uvicorn app:app --reload
```

Then open http://127.0.0.1:8000/docs and use "Try it out". On Mac/Linux, activate with `source .venv/bin/activate`.

### API usage

```
curl -X POST http://127.0.0.1:8000/predict ^
     -F "file=@sample.jpeg;type=image/jpeg"

curl -X POST http://127.0.0.1:8000/analyze ^
     -F "file=@sample.jpeg;type=image/jpeg" ^
     -F "note=Fever and chills since Monday. Guardian 0300-1234567, CNIC 35202-1234567-1"
```

(On Mac/Linux use `\` instead of `^`.)

Results I observed on my sample image:

- `/health`: `status: ok`, `model_loaded: true`, classes `["Parasitized", "Uninfected"]`, positive `Parasitized`.
- `/predict`: label `Parasitized`, probability about 0.9999 for Parasitized.
- `/analyze`: `note_masked` shows `[PHONE]` and `[CNIC]` instead of the real numbers, `urgent_terms` contains `fever` and `chills`, and `decision` is `REFER`. The digits `1234567` do not appear anywhere in the response.

#### Error handling

| Bad input | Status | Response |
|---|---|---|
| A non-image file (`app.py` sent as `text/plain`) | 415 | `{"detail":"Upload a JPEG or PNG image"}` |
| An image larger than 5 MB | 413 | `{"detail":"Image larger than 5 MB"}` |
| A file that is not a readable image | 400 | `{"detail":"File is not a readable image"}` |
| Missing file | 422 | validation error from FastAPI |
| A note longer than 2,000 characters | 422 | `{"detail":"Note longer than 2,000 characters"}` |

**One bad input beyond the tests:** I sent a 2,001-character note to `/analyze`. The API returned `{"detail":"Note longer than 2,000 characters"}` with status 422, and no stack trace. I also sent a 6 MB file to `/predict` and got `{"detail":"Image larger than 5 MB"}`.

### Tests

```
pytest -q test_api.py
........                                  [100%]
8 passed, 1 warning in 32.30s
```

The 8 tests cover: health, response schema and probabilities summing to 1, parity between the API and calling the model directly, wrong file type (415), too-large file (413), corrupt image (400), missing file (422), and PII masking in `/analyze` (no part of the phone number or CNIC appears in the response). The single warning is a deprecation notice from a library and does not affect the results.

### Latency and capacity

Benchmark: 50 requests to `/predict` against a running server (`python bench.py`).

| Item | Value |
|---|---|
| Hardware | Intel Core i5-6300U @ 2.40 GHz (2 cores), 7.9 GB RAM, CPU only |
| p50 | 271 ms |
| p95 | 660 ms |
| max | 1343 ms |
| Throughput, 1 worker | about 3.0 requests/s |

**Little's law.** L = λW with L = 1 for one worker and W = p50 = 0.271 s, so λ = 1 / 0.271 ≈ 3.7 requests/s. Over a day that is about 3.7 x 86,400 ≈ **319,000 requests per day** if traffic were perfectly even. At the measured 3.0 requests/s it is about 259,000 per day. Real traffic comes in bursts, so safe capacity is lower.

### Check Your Understanding

1. **A client uploads a PDF to /predict. Which status code, and why not 500?** 415 Unsupported Media Type. The client sent the wrong kind of file, which is a client error (4xx). A 500 means the server crashed from a bug and could leak a stack trace. The API checks the content type first and returns a clean message.
2. **Why is the model loaded in lifespan rather than inside the endpoint?** Loading the checkpoint and rebuilding the network takes seconds. In `lifespan` it happens once at startup, so each request only runs the fast forward pass. Loading inside the endpoint would add that delay to every request and waste memory.
3. **p50 = 40 ms and p95 = 400 ms. What does that tell you, and what would you check?** Most requests are fast, but about 1 in 20 is ten times slower: a tail-latency problem that the average hides. I would check for cold starts, background processes or CPU throttling, blocking of the event loop, requests queuing together, and unusually large inputs. In my own run (p50 271 ms, p95 660 ms) the likely causes are a low-power 2-core laptop CPU and other programs running at the same time.
4. **500,000 requests a day, peak 5 times the average, 25 requests per second per worker. How many workers?** Average = 500,000 / 86,400 ≈ 5.8 requests/s. Peak = 5 x 5.8 ≈ 29 requests/s. Workers = 29 / 25 ≈ 1.16, so 2 workers.
5. **Why can model inference inside an async def endpoint slow down other requests?** An async server runs on one event loop that switches between requests only at an `await`. Model inference is heavy CPU work with no `await`, so the loop is blocked and other requests, even `/health`, wait. `run_in_threadpool` moves the work to a thread so the loop stays free.
6. **How many communication channels does a 4-person team have?** n(n - 1) / 2 = 4 x 3 / 2 = 6.

### Privacy

`/analyze` masks CNIC numbers and mobile numbers before any other processing, so the original digits never appear in the response. Masking runs first, and a test checks the whole response for leaks.

### Reflection

Building this API showed me that a model that works in a notebook is only the start of a working service. My first PyTorch install timed out on a slow connection, and I fixed it by retrying with a longer timeout and more retries. Later, an indentation error stopped the server from starting after I saved `app.py` in the wrong format, which taught me that the server needs the file as exact plain text. The benchmark surprised me most: on my i5-6300U laptop the median was 271 ms but p95 was 660 ms, so the average hides slow requests and capacity planning should use the tail, not the mean. Testing bad inputs (a text file, a 6 MB file and a 2,001-character note) was reassuring, because each got a clean error code with no stack trace, and the PII test showed that the phone number and CNIC never appear in a response. If I continued this project, I would add an API key, deploy it publicly, and tune the threshold so that Parasitized cases are rarely missed.

### Limits

This is a student project and a decision-support demo. It must not be used to diagnose patients or replace a laboratory professional.
