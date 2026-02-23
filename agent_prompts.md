# LLM Agent Prompts — Tatweer AI/ML Technical Assignment

---

## Challenge Option 1: Computer Vision — Near-Miss Incident Detection

---

### Task 1.1 — Environment Setup & Video Ingestion

```
You are an AI/ML engineer working on a computer vision pipeline for near-miss traffic incident detection.

Your first task is to set up the Python environment and load the source video for processing.

**Inputs:**
- Video URL: https://www.youtube.com/watch?v=r86kxxU-LUY
- Runtime constraint: Google Colab free tier (CPU only) or local CPU

**Steps to implement:**
1. Install all required dependencies. The stack must be CPU-compatible and open-source. Recommended packages: `opencv-python`, `torch` (CPU build), `torchvision`, `ultralytics` (YOLOv8), `numpy`, `matplotlib`, `pandas`, `yt-dlp` (for downloading the YouTube video), `Pillow`, `scipy`.
2. Download the video from YouTube using `yt-dlp` at 720p or lower (to keep file size manageable on Colab).
3. Load the video with OpenCV (`cv2.VideoCapture`). Extract and print metadata: total frame count, FPS, width, height, duration in seconds.
4. Write a helper function `sample_frames(video_path, n=5)` that reads `n` evenly spaced frames from the video and displays them in a matplotlib grid. Use this to visually confirm the video loaded correctly.
5. Create a `requirements.txt` listing all installed packages and their versions.
6. Structure your Jupyter notebook with clear markdown cells separating each section: **Setup**, **Video Loading**, **Preprocessing**.

**Output expected:**
- A fully executable notebook cell block that installs dependencies, downloads the video, prints video metadata, and displays sample frames.
- `requirements.txt` file.

**Constraints:**
- Do NOT use any paid APIs or cloud GPU services.
- All code must run end-to-end without manual intervention after cell execution.
```

---

### Task 1.2 — Object Detection with a Pre-Trained Model

```
You are building the object detection component of a near-miss incident detection system for traffic video analysis.

**Context:**
- You have a traffic video already loaded via OpenCV (from a prior step).
- Runtime is CPU-only (Google Colab free tier or local machine).
- You must detect vehicles and pedestrians in each video frame.

**Task:**
Implement a frame-by-frame object detection pipeline using a pre-trained model.

**Detailed Requirements:**

1. **Model Selection:**
   - Use YOLOv8n (nano) from the `ultralytics` library — it is the lightest YOLO variant suitable for CPU inference.
   - Alternatively, you may use YOLOv5n or a MobileNet-SSD if you justify your choice.
   - Filter detections to only the following COCO classes: `car`, `truck`, `bus`, `motorcycle`, `bicycle`, `person`. Map these to two internal categories: `vehicle` and `pedestrian`.

2. **Detection Pipeline:**
   - Write a function `detect_objects(frame, model, conf_threshold=0.4)` that:
     - Accepts a BGR frame (numpy array from OpenCV).
     - Runs inference with the chosen model.
     - Returns a list of detection dicts, each with keys: `{ "id": None, "class": str, "bbox": [x1, y1, x2, y2], "confidence": float, "center": (cx, cy) }`.
   - Apply non-maximum suppression (NMS) if not already handled by the model wrapper.

3. **Performance Optimization:**
   - Resize frames to 640×640 before inference (standard YOLO input size).
   - Process every other frame (frame stride = 2) to reduce computation while maintaining reasonable temporal resolution.
   - Log average inference time per frame.

4. **Validation:**
   - Run detection on 10 sample frames.
   - Display annotated frames (bounding boxes + class labels + confidence scores) in a matplotlib grid.
   - Confirm that vehicles and pedestrians are being detected correctly.

**Output:**
- A `detect_objects()` function ready to be integrated into the full video processing loop.
- Sample annotated frames shown in the notebook.
- A printed summary: average inference time per frame, number of detections per class across the sample.

**Constraints:**
- Total inference time for the full 2–3 minute video must be under 5 minutes on CPU.
- Do not use paid or proprietary models.
```

---

### Task 1.3 — Multi-Object Tracking Across Frames

```
You are implementing the object tracking module for a near-miss detection pipeline. Detections from a YOLO model are already available per frame. Now you must associate detections across frames to maintain consistent object identities over time.

**Context:**
- Per-frame detections are provided as lists of dicts: `{ "class": str, "bbox": [x1, y1, x2, y2], "confidence": float, "center": (cx, cy) }`.
- Video is processed at a stride of 2 frames, so the effective FPS is roughly half the original.

**Task:**
Implement a centroid-based tracker (or integrate SORT/DeepSORT if feasible on CPU).

**Option A — Centroid Tracker (Recommended for simplicity):**
1. Implement a `CentroidTracker` class with:
   - `register(centroid)`: assign a new unique integer ID to a new object.
   - `deregister(object_id)`: remove objects that have been missing for more than `max_disappeared=10` frames.
   - `update(detections)`: accepts the list of detection dicts for the current frame, matches them to existing tracked objects using Euclidean distance between centroids (use `scipy.spatial.distance.cdist`), updates positions, and returns an `OrderedDict` mapping `object_id -> { "class", "bbox", "center", "confidence" }`.
2. Handle the case where there are more detections than existing tracked objects (register new objects) and more tracked objects than detections (increment disappeared counter).

**Option B — SORT Integration (if preferred):**
1. Install `sort-tracker` or implement SORT using Kalman filter + Hungarian algorithm.
2. Wrap SORT so its output format matches the same `OrderedDict` structure as Option A.

**Tracking Requirements:**
- Each tracked object must maintain a trajectory: a list of (frame_index, center_x, center_y) tuples — used later for speed/trajectory analysis.
- Store class label per object ID (use the most frequently detected class for that ID if it fluctuates).

**Validation:**
- Run the tracker on the first 300 frames of the video.
- Plot the trajectories of all tracked objects on a single canvas (use `matplotlib`, one color per object ID).
- Print: total unique IDs assigned, average track length (frames), number of objects active at peak frame.

**Output:**
- A `CentroidTracker` class (or SORT wrapper) ready for integration.
- A trajectory visualization plot.
- Printed tracking statistics.
```

---

### Task 1.4 — Near-Miss Detection Logic

```
You are implementing the core near-miss detection engine for a traffic safety system. Object tracking data (IDs, positions, trajectories, classes) is available for each frame.

**Definition of a Near-Miss:**
A near-miss event occurs when two tracked objects satisfy at least TWO of the following criteria simultaneously:
1. Their bounding boxes overlap or their centroid distance falls below a proximity threshold.
2. Their estimated Time-To-Collision (TTC) is below a critical threshold.
3. Their relative velocity indicates they are closing in on each other rapidly.

**Task:**
Implement a `NearMissDetector` class and a risk scoring system.

**Detailed Implementation:**

1. **Proximity Check:**
   - Compute pixel distance between centroids of every pair of tracked objects in each frame.
   - Flag a pair as "proximate" if distance < `proximity_threshold` (tune empirically; suggest starting at 100 pixels, adjustable as a parameter).
   - Also compute IoU between bounding boxes as a secondary proximity signal.

2. **Speed & Trajectory Estimation:**
   - For each tracked object, compute instantaneous speed as the Euclidean distance between centroid positions in consecutive processed frames, divided by the time delta (1 / effective_FPS seconds).
   - Estimate heading direction as the angle (degrees) of the velocity vector.

3. **Time-To-Collision (TTC) Estimation:**
   - For each proximate pair, compute TTC = current_distance / closing_speed, where closing_speed is the component of relative velocity along the line connecting the two centroids.
   - Flag if TTC < `ttc_threshold` (suggest 2.0 seconds as default).
   - Handle edge cases: if objects are moving apart, TTC is infinity (no collision risk).

4. **Risk Scoring:**
   - Assign a risk score (0.0–1.0) per detected near-miss event using:
     - `score = (1 - normalized_distance) * 0.4 + (1 - normalized_TTC) * 0.4 + speed_factor * 0.2`
   - Map score to risk level: High (≥0.7), Medium (0.4–0.7), Low (<0.4).

5. **Event Logging:**
   - For each near-miss event, log: `{ "frame_index": int, "timestamp_sec": float, "object_id_1": int, "object_id_2": int, "class_1": str, "class_2": str, "distance_px": float, "ttc_sec": float, "risk_score": float, "risk_level": str }`.
   - Apply a debounce mechanism: the same pair of objects cannot trigger a new event within 30 frames of the last event.

**Output:**
- `NearMissDetector` class with a `process_frame(frame_index, tracked_objects)` method.
- A pandas DataFrame of all logged near-miss events, sorted by timestamp.
- Print: total events detected, breakdown by risk level (High/Medium/Low), top 3 highest-risk events.
```

---

### Task 1.5 — Visualization, Annotation & Summary Report

```
You are building the output and reporting layer of a near-miss detection system. You have:
- The original video file.
- Per-frame tracking data (object IDs, bounding boxes, classes).
- A DataFrame of detected near-miss events with timestamps and risk scores.

**Task:**
Produce an annotated output video and a structured analysis report.

**Part A — Annotated Video:**
1. Use OpenCV's `VideoWriter` to create an output video at the same FPS and resolution as the input.
2. For each frame:
   - Draw bounding boxes for all tracked objects. Color by class: vehicles = blue, pedestrians = green.
   - Display object ID and class label above each bounding box.
   - For any near-miss event active in this frame, draw bounding boxes in RED for the involved objects and display the risk level (e.g., "HIGH RISK") in large red text at the top of the frame.
   - Overlay a frame counter and timestamp (MM:SS) in the top-left corner.
3. Save the output as `outputs/annotated_video.mp4`.

**Part B — Dashboard Visualizations (matplotlib/seaborn):**
Create the following plots and save each as a PNG in `outputs/`:

1. **Timeline Plot** (`timeline.png`): A horizontal scatter/event plot showing near-miss events on the video timeline (x-axis = time in seconds, y-axis = risk level). Color-code by risk: red = High, orange = Medium, yellow = Low.

2. **Risk Distribution Pie Chart** (`risk_distribution.png`): Pie chart showing the proportion of High / Medium / Low events.

3. **Object Activity Heatmap** (`heatmap.png`): Aggregate all centroid positions of all tracked objects across all frames onto a 2D grid. Use `numpy.histogram2d` and display as a heatmap overlaid on the first video frame. This shows where activity is densest.

4. **Near-Miss Frequency Bar Chart** (`frequency.png`): Bin the video into 15-second intervals. For each bin, count the number of near-miss events. Display as a bar chart.

**Part C — Summary Report:**
Generate an HTML report (`outputs/report.html`) using basic Python string formatting or `jinja2` (if available). The report must include:
- Video metadata (duration, FPS, resolution).
- Total objects tracked, total unique IDs.
- Total near-miss events: count by risk level.
- Top 5 highest-risk events (timestamp, object classes, TTC, risk score).
- All four visualization images embedded inline.
- A section titled "Limitations & Assumptions" with at least 3 bullet points discussing known shortcomings of the approach (e.g., pixel-space distance not accounting for perspective distortion, lack of depth estimation, static camera assumption).

**Output:**
- `outputs/annotated_video.mp4`
- `outputs/timeline.png`, `risk_distribution.png`, `heatmap.png`, `frequency.png`
- `outputs/report.html`
```

---

### Task 1.6 — (Bonus) False Positive Reduction

```
You are adding a false-positive filtering layer to an existing near-miss detection system. The system currently flags many events that are not genuine near-misses (e.g., stationary vehicles parked close together, noise in tracking).

**Task:**
Implement a post-processing filter that reduces false positives before events are logged.

**Filters to implement (apply all):**

1. **Stationary Object Filter:**
   - Compute the average speed of each object over the last 15 frames.
   - If BOTH objects in a detected pair have average speed < 5 px/frame, discard the event (parked vehicles touching each other are not a near-miss).

2. **Minimum Duration Filter:**
   - Only confirm a near-miss event if the proximity condition is met for at least 5 consecutive processed frames. Implement this as a "confirmation buffer" per object pair.

3. **Direction-of-Travel Filter:**
   - If both objects are travelling in roughly the same direction (heading angle difference < 30°) AND their closing speed is near zero, discard the event (vehicles travelling in convoy are not a near-miss).

4. **Confidence Gate:**
   - Discard any event where either of the two involved detections had a YOLO confidence score < 0.5 at the frame of the event.

**Evaluation:**
- Re-run the full pipeline with filters enabled vs. disabled.
- Print a table comparing: total events before/after filtering, breakdown by risk level before/after.
- Manually review 5 filtered-out events and 5 kept events. In the notebook, display the annotated frames for those events and write a markdown cell assessing whether the filter made the right decision.

**Output:**
- Updated `NearMissDetector` class incorporating all filters.
- Comparison table (before vs. after filtering).
- Manual review section with annotated frames and written assessment.
```

---

### Task 1.7 — (Bonus) Real-Time Optimization

```
You are tasked with optimizing the near-miss detection pipeline for near-real-time performance on CPU.

**Target:** Achieve ≥ 15 FPS end-to-end processing throughput on a standard CPU.

**Optimization strategies to implement and benchmark:**

1. **Model Quantization:**
   - Export YOLOv8n to ONNX format using `model.export(format='onnx')`.
   - Run inference using `onnxruntime` (CPU provider) instead of PyTorch.
   - Measure FPS before and after.

2. **Input Resolution Reduction:**
   - Test inference at 320×320 vs. 640×640.
   - Measure detection quality (% of objects still detected) vs. FPS trade-off.

3. **Frame Skipping Strategy:**
   - Test frame strides of 1, 2, 3, 4.
   - Plot FPS vs. number of near-miss events missed (compare to stride=1 as ground truth).

4. **Threading:**
   - Implement a producer-consumer architecture using Python `threading` and a `Queue`:
     - Producer thread: reads frames from video.
     - Consumer thread: runs inference.
   - Measure FPS improvement.

5. **Results:**
   - Produce a benchmarking table: `{ "optimization": str, "fps": float, "missed_events_%": float }`.
   - Recommend the best configuration for a production deployment based on your findings.
   - Write a markdown cell explaining the trade-offs.

**Constraint:** All optimizations must remain CPU-only and open-source.
```

---
---

## Challenge Option 2: Generative AI — Fine-Tuning a Small Language Model

---

### Task 2.1 — Task Selection, Dataset Design & Curation

```
You are an ML engineer preparing a fine-tuning dataset for a small language model (<1B parameters).

**Your chosen task:** Domain-specific Q&A for technical support (IT/software troubleshooting).
*(You may substitute a different domain — e.g., medical triage Q&A, legal clause simplification, Python code generation for a specific library — but the instructions below must be adapted accordingly.)*

**Task:**
Design, curate, and preprocess a high-quality instruction-response dataset.

**Dataset Requirements:**
- Minimum 200 examples, target 300–500 for better results.
- Format: instruction-following pairs. Each example must be a JSON object:
  ```json
  {
    "instruction": "A clear, specific question or task description.",
    "input": "",  // optional additional context; leave empty string if not needed
    "output": "A correct, complete, helpful answer."
  }
  ```
- Split: 80% train, 10% validation, 10% test.

**Curation Steps:**

1. **Source Selection:**
   - Use at least 2 of the following sources:
     a. Publicly available Q&A datasets from HuggingFace Hub (e.g., `StackExchange`, `dolly-15k`, `Open-Platypus`). Filter for your domain.
     b. Synthetic generation: write a prompt that asks GPT-4 or Claude to generate diverse Q&A pairs for your domain. Document the prompt used.
     c. Manual curation: write 20–30 high-quality examples yourself to serve as "seed" data.
   - Document each source: name, URL/reference, number of examples taken.

2. **Quality Filtering:**
   - Remove examples where `output` is fewer than 20 words or more than 500 words.
   - Remove duplicates using exact-match deduplication on the `instruction` field.
   - Remove examples that contain personally identifiable information (PII) patterns (email addresses, phone numbers) using regex.
   - Flag and remove any examples where the output appears to be cut off (ends mid-sentence without punctuation).

3. **Formatting:**
   - Normalize whitespace (strip leading/trailing spaces, collapse multiple spaces).
   - Ensure instructions end with a question mark or imperative verb (e.g., "Explain...", "What is...").
   - Save the curated dataset as three JSONL files: `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`.

4. **Data Card:**
   Write a markdown section titled "Data Card" in the notebook covering:
   - Task description and intended use.
   - Sources and their licenses.
   - Dataset statistics: example count per split, average instruction length, average output length, vocabulary coverage.
   - Known biases or limitations.

**Output:**
- `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`.
- Printed dataset statistics (counts, average lengths, unique token estimate).
- Data Card markdown section.
```

---

### Task 2.2 — Model Selection & LoRA Fine-Tuning Setup

```
You are implementing parameter-efficient fine-tuning (PEFT) using LoRA on a small language model.

**Constraints:**
- Model size: < 1B parameters.
- Must use LoRA or QLoRA (no full fine-tuning).
- Training must complete in < 2 hours on Google Colab free tier (CPU or T4 GPU if available).
- Use `unsloth`, `peft`, or `transformers` + `trl`.

**Task:**
Set up the model, tokenizer, LoRA configuration, and training pipeline.

**Step-by-Step Implementation:**

1. **Model Selection (choose one and justify):**
   - `HuggingFaceTB/SmolLM-135M-Instruct` — smallest, fastest, good for CPU.
   - `Qwen/Qwen2-0.5B-Instruct` — slightly larger, strong instruction following.
   - `microsoft/phi-2` — 2.7B (borderline; only use if GPU available and you justify why quality warrants the size).
   - In a markdown cell, explain your choice based on: task requirements, parameter count, available compute, and prior benchmark performance on instruction-following tasks.

2. **Tokenizer Setup:**
   - Load the tokenizer with `use_fast=True`.
   - Set `padding_side='right'` for causal LMs.
   - Define a prompt template function `format_prompt(example)` that converts each dataset example to a single string in the model's chat/instruction format. For example:
     ```
     ### Instruction:
     {instruction}

     ### Input:
     {input}

     ### Response:
     {output}<EOS>
     ```
   - Tokenize the dataset using `datasets.map()`, truncating to `max_length=512`. Do NOT pad during tokenization (use dynamic padding in the DataCollator instead).

3. **LoRA Configuration:**
   Configure `peft.LoraConfig` with the following (document your reasoning for each):
   - `r` (rank): start with 16. Explain: higher rank = more capacity but more parameters.
   - `lora_alpha`: set to 32 (2× rank is a common heuristic).
   - `target_modules`: identify the correct module names for your chosen model (e.g., `["q_proj", "v_proj"]` for attention layers). Print the model architecture to find them.
   - `lora_dropout`: 0.05.
   - `bias`: "none".
   - `task_type`: "CAUSAL_LM".
   - Print the number of trainable parameters vs. total parameters, and the percentage.

4. **Training Configuration (`transformers.TrainingArguments`):**
   Justify each hyperparameter in an inline comment:
   - `num_train_epochs`: 3
   - `per_device_train_batch_size`: 4 (adjust down to 2 if OOM)
   - `gradient_accumulation_steps`: 4 (effective batch size = 16)
   - `learning_rate`: 2e-4
   - `lr_scheduler_type`: "cosine"
   - `warmup_ratio`: 0.03
   - `fp16`: True if GPU, False if CPU
   - `logging_steps`: 10
   - `evaluation_strategy`: "steps", `eval_steps`: 50
   - `save_strategy`: "steps", `save_steps`: 100, `save_total_limit`: 2
   - `load_best_model_at_end`: True
   - `metric_for_best_model`: "eval_loss"
   - `early_stopping_patience`: 3 (via `EarlyStoppingCallback`)

5. **Trainer Setup:**
   - Use `trl.SFTTrainer` or `transformers.Trainer` with a `DataCollatorForLanguageModeling` (mlm=False).
   - Attach `EarlyStoppingCallback`.
   - Start training and log output.

**Output:**
- Complete, runnable training notebook cells.
- Printed trainable parameter count and percentage.
- Markdown cell documenting all hyperparameter choices and rationale.
```

---

### Task 2.3 — Training Execution, Monitoring & Checkpoint Management

```
You are running and monitoring the LoRA fine-tuning training loop for a small language model.

**Context:**
- Training setup (model, LoRA config, TrainingArguments, SFTTrainer) is already configured.
- Dataset splits (train/val) are loaded and tokenized.

**Task:**
Execute training, monitor metrics, and manage checkpoints.

**Implementation:**

1. **Training Execution:**
   - Call `trainer.train()`.
   - Capture all logged metrics (loss, eval_loss, learning_rate) via the trainer's log history.
   - After training completes, print: final training loss, final validation loss, number of epochs completed (may be fewer than max if early stopping triggered), total training time.

2. **Loss Curve Plotting:**
   - Parse `trainer.state.log_history` to extract training loss and validation loss at each logging step.
   - Plot both curves on the same matplotlib figure (x-axis = training step, y-axis = loss).
   - Add a vertical dashed line where the best checkpoint was saved (lowest eval loss).
   - Save as `outputs/loss_curves.png`.
   - In a markdown cell below the plot, interpret the curves: Is the model converging? Is there evidence of overfitting (train loss decreasing while eval loss increases)? Did early stopping activate?

3. **Checkpoint Management:**
   - The best checkpoint is saved automatically by the trainer.
   - Load the best checkpoint: `model = PeftModel.from_pretrained(base_model, best_checkpoint_path)`.
   - Verify the loaded model produces coherent output by running inference on one validation example.

4. **Saving the Final Model:**
   - Save the LoRA adapter weights: `model.save_pretrained("outputs/lora_adapter")`.
   - Save the tokenizer: `tokenizer.save_pretrained("outputs/lora_adapter")`.
   - Do NOT save the full merged model (too large for submission); save adapters only.

5. **Overfitting Indicators Checklist (markdown cell):**
   Write a markdown checklist assessing each indicator:
   - [ ] Training loss is significantly lower than validation loss at end of training.
   - [ ] Validation loss stopped improving before training loss.
   - [ ] Generated outputs on validation set show memorized training examples.
   - [ ] BLEU/ROUGE scores on validation are much higher than on the held-out test set.

**Output:**
- `outputs/loss_curves.png`
- `outputs/lora_adapter/` directory (LoRA adapter weights + tokenizer)
- Printed training summary statistics
- Markdown interpretation of training dynamics
```

---

### Task 2.4 — Quantitative Evaluation & Base Model Comparison

```
You are evaluating the performance of a fine-tuned small language model and comparing it against its base (pre-fine-tuning) counterpart.

**Context:**
- Fine-tuned model (LoRA adapter) is saved at `outputs/lora_adapter/`.
- Base model (same architecture, no fine-tuning) is available from HuggingFace Hub.
- Test set is at `data/test.jsonl` (unseen during training).

**Task:**
Compute quantitative metrics for both models and compare.

**Metrics to Implement:**

1. **Perplexity:**
   - For each model (base and fine-tuned), compute perplexity on the test set.
   - Perplexity = exp(average negative log-likelihood per token).
   - Use the model's `forward()` with `labels` to get cross-entropy loss, then exponentiate.
   - Report perplexity per model. Lower = better.

2. **ROUGE Scores (ROUGE-1, ROUGE-2, ROUGE-L):**
   - Generate a response for each test example using greedy decoding (`model.generate()` with `max_new_tokens=200`, `do_sample=False`).
   - Compare generated text to ground-truth `output` using the `rouge_score` library.
   - Report mean ROUGE-1, ROUGE-2, ROUGE-L F1 scores across all test examples for both models.

3. **BLEU Score:**
   - Use `nltk.translate.bleu_score.corpus_bleu()`.
   - Report corpus-level BLEU for both models.

4. **Task-Specific Metric (choose one based on your task):**
   - For Q&A tasks: exact-match accuracy (does the generated answer contain the key entity/fact from the ground truth?).
   - For code generation: compilation success rate (does the generated code execute without syntax errors? Use `ast.parse()` for Python).
   - For style transfer: a simple classifier-based fluency score (optional).

5. **Results Table:**
   Create a pandas DataFrame with columns: `Metric | Base Model | Fine-Tuned Model | Improvement (%)`.
   Print and display this table in the notebook.

6. **Statistical Note:**
   - Compute 95% confidence intervals for ROUGE-L using bootstrap resampling (sample 1000 bootstrap subsets of test examples, compute ROUGE-L each time, report 2.5th and 97.5th percentile).

**Output:**
- Results comparison table (base vs. fine-tuned).
- `outputs/evaluation_results.csv` containing all per-example scores.
- Markdown cell summarizing findings: which metric improved the most? Is the improvement practically significant? What does this tell you about what the fine-tuning learned?
```

---

### Task 2.5 — Qualitative Analysis & Error Analysis

```
You are conducting qualitative evaluation of a fine-tuned language model. Quantitative metrics are already computed; now you need human-readable analysis.

**Context:**
- Fine-tuned model and base model are both loaded.
- Test set is available at `data/test.jsonl`.
- You have per-example ROUGE-L scores from the previous evaluation step.

**Task:**
Produce qualitative output examples and a structured error analysis.

**Part A — Good Example Showcase (5 examples):**
1. Select the 5 test examples with the highest ROUGE-L score for the fine-tuned model.
2. For each example, display in a formatted notebook cell:
   - **Instruction:** (the input prompt)
   - **Ground Truth:** (the expected output)
   - **Base Model Output:** (generated by the un-fine-tuned model)
   - **Fine-Tuned Model Output:** (generated by the fine-tuned model)
   - **ROUGE-L score:** (fine-tuned)
3. Write a 2–3 sentence commentary for each explaining what specifically improved.

**Part B — Failure Case Analysis (5 examples):**
1. Select the 5 test examples with the lowest ROUGE-L score for the fine-tuned model.
2. Display in the same format as Part A.
3. For each failure, write a 2–3 sentence analysis categorizing the error type from the following taxonomy:
   - **Hallucination:** model generates plausible-sounding but factually wrong content.
   - **Incomplete response:** model stops too early or misses key parts of the answer.
   - **Off-topic:** model misunderstands the instruction and answers a different question.
   - **Formatting failure:** correct content but wrong structure (e.g., missing bullet points, wrong code block).
   - **Overfitting artifact:** model appears to be reciting training data verbatim.
   - **Other:** describe.

**Part C — Systematic Error Patterns:**
1. Tally the error types from Part B across all low-scoring examples (not just the 5 shown).
2. Create a bar chart of error type frequencies (`outputs/error_analysis.png`).
3. Write a markdown section "Root Cause Hypotheses" with at least 3 hypotheses explaining why these errors occur and what training/data changes could address them.

**Output:**
- Formatted display of 5 good + 5 bad examples with commentary.
- `outputs/error_analysis.png`
- Markdown "Root Cause Hypotheses" section.
```

---

### Task 2.6 — Model Card

```
You are writing a Model Card for a fine-tuned small language model. Model cards are standardized documentation that communicate what a model does, how it was trained, its performance, and its limitations.

**Task:**
Write a complete `MODEL_CARD.md` file following the HuggingFace model card standard.

**Required Sections:**

---

# Model Card: [Your Model Name]

## Model Details
- **Base Model:** [e.g., HuggingFaceTB/SmolLM-135M-Instruct]
- **Fine-Tuning Method:** LoRA (Low-Rank Adaptation) via PEFT
- **LoRA Config:** r=[value], alpha=[value], target_modules=[list], dropout=[value]
- **Model Size:** [parameter count] parameters total; [trainable parameter count] trainable
- **Training Date:** [date]
- **Framework:** Transformers [version], PEFT [version], PyTorch [version]

## Intended Use
- **Primary Use Case:** [Describe the specific task, e.g., "Answering technical IT support questions in English"]
- **Intended Users:** [e.g., help desk automation systems, developers building support chatbots]
- **Out-of-Scope Uses:** [e.g., medical advice, legal counsel, any high-stakes decision-making]

## Training Data
- **Dataset:** [Name and description]
- **Size:** [N] training examples, [N] validation, [N] test
- **Sources:** [List with licenses]
- **Preprocessing:** [Summary of filtering and formatting steps]

## Training Procedure
- **Hardware:** [e.g., Google Colab T4 GPU / CPU]
- **Hyperparameters:** Learning rate, batch size, epochs, warmup ratio (table format)
- **Training Time:** [Duration]
- **Early Stopping:** [Yes/No, patience value]

## Evaluation Results
| Metric       | Base Model | Fine-Tuned Model | Improvement |
|---|---|---|---|
| Perplexity   | ...        | ...              | ...         |
| ROUGE-1      | ...        | ...              | ...         |
| ROUGE-2      | ...        | ...              | ...         |
| ROUGE-L      | ...        | ...              | ...         |
| BLEU         | ...        | ...              | ...         |

## Limitations & Biases
- [Minimum 5 bullet points covering: data biases, domain coverage gaps, hallucination tendency, language limitations, out-of-distribution behavior]

## How to Use
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("BASE_MODEL_ID")
model = PeftModel.from_pretrained(base_model, "path/to/lora_adapter")
tokenizer = AutoTokenizer.from_pretrained("path/to/lora_adapter")

prompt = "### Instruction:\nYour question here\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Citation
[If applicable]

---

**Output:**
- `MODEL_CARD.md` file in the submission root directory.
- All numeric values in the evaluation table must be filled in from actual training and evaluation runs (no placeholder values).
```

---

### Task 2.7 — (Bonus) Catastrophic Forgetting Analysis

```
You are investigating whether fine-tuning a small language model on a domain-specific task has degraded its general language capabilities — a phenomenon known as "catastrophic forgetting."

**Task:**
Design and run a suite of general-capability tests on both the base model and the fine-tuned model, then compare.

**Tests to Implement:**

1. **General Q&A (5 questions):**
   - Use 5 factual questions outside your fine-tuning domain (e.g., geography, science, history).
   - Example: "What is the capital of France?", "Explain why the sky is blue in one sentence."
   - Score each answer 0 or 1 for factual correctness (manual evaluation).

2. **Instruction Following (5 tasks):**
   - Give 5 simple instruction-following prompts unrelated to your domain (e.g., "List 3 types of fruit", "Write a haiku about rain", "Translate 'hello' to Spanish").
   - Score each 0 or 1 for whether the model followed the instruction format correctly.

3. **Language Fluency (perplexity on general corpus):**
   - Download 100 random sentences from the WikiText-103 test set.
   - Compute perplexity of both models on these sentences.
   - A significantly higher perplexity in the fine-tuned model indicates degraded general language modeling.

4. **Mathematical Reasoning (5 problems):**
   - Give 5 simple arithmetic or logic problems (e.g., "What is 17 × 8?", "If A > B and B > C, is A > C?").
   - Score correctness.

**Results:**
- Create a table: `{ "Test Category": str, "Base Model Score": float, "Fine-Tuned Score": float, "Degradation %": float }`.
- Write a markdown section "Catastrophic Forgetting Assessment" that:
  - Summarizes which capabilities were preserved and which degraded.
  - Hypothesizes why (e.g., small model = less redundancy, domain data overwrote general patterns).
  - Suggests mitigation strategies (e.g., replay buffer, multi-task training, lower learning rate).

**Output:**
- Comparison table of general-capability scores.
- Markdown "Catastrophic Forgetting Assessment" section.
```

---

### Task 2.8 — (Bonus) Prompt Engineering Baseline Comparison

```
You are adding a baseline comparison to assess whether fine-tuning was actually necessary, or if clever prompt engineering of the base model achieves comparable results.

**Task:**
Implement 3 prompting strategies on the base model and compare their performance against the fine-tuned model.

**Prompting Strategies:**

1. **Zero-Shot:**
   - Use a simple system prompt: "You are a helpful expert assistant. Answer the following question accurately and concisely."
   - Append the instruction directly. No examples provided.

2. **Few-Shot (3 examples):**
   - Prepend 3 high-quality examples from the training set (formatted as instruction-response pairs) before each test query.
   - Use a consistent separator between examples.

3. **Chain-of-Thought (CoT):**
   - Add "Let's think step by step." to the end of each instruction before the model generates its response.
   - Evaluate whether this improves structured/reasoning tasks.

**Evaluation:**
- Run all 3 prompting strategies + the fine-tuned model on the full test set.
- Compute ROUGE-L and BLEU for each.
- Create a bar chart comparing all 4 approaches (`outputs/baseline_comparison.png`).

**Analysis (markdown cell):**
- When does fine-tuning outperform few-shot? (Likely: short, consistent format; domain vocabulary; speed at inference due to shorter prompts.)
- When does few-shot match or beat fine-tuning? (Likely: very small training set; base model already capable in the domain.)
- What are the trade-offs between fine-tuning and prompting in production? (Cost, latency, maintainability, catastrophic forgetting risk.)

**Output:**
- Code for all 3 prompting strategies.
- `outputs/baseline_comparison.png`
- Markdown analysis section.
```

---

## Submission Assembly Prompt

```
You are finalizing the submission package for the Tatweer AI/ML Engineering take-home challenge.

**Task:**
Organize all outputs into the required directory structure and create a README.

**Required Directory Structure:**
```
your-name-challenge/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_setup_and_data.ipynb
│   ├── 02_model_training.ipynb      (Challenge 2 only)
│   ├── 02_detection_tracking.ipynb  (Challenge 1 only)
│   ├── 03_evaluation.ipynb
│   └── 04_bonus.ipynb               (if bonus tasks completed)
├── data/
│   ├── train.jsonl                  (Challenge 2) or  raw_video.mp4 (Challenge 1)
│   ├── val.jsonl
│   └── test.jsonl
├── outputs/
│   ├── annotated_video.mp4          (Challenge 1)
│   ├── report.html                  (Challenge 1)
│   ├── lora_adapter/               (Challenge 2)
│   ├── loss_curves.png             (Challenge 2)
│   └── [all other generated PNGs]
├── MODEL_CARD.md                    (Challenge 2 only)
└── demo_video.mp4
```

**README.md must include:**
1. **Project Title** and chosen challenge name.
2. **Setup Instructions** — step-by-step commands to reproduce the environment (`pip install -r requirements.txt` or Colab link).
3. **How to Run** — order in which notebooks should be executed; any manual steps (e.g., placing the video file).
4. **Results Summary** — 3–5 bullet points with key findings or metric highlights.
5. **Bonus Tasks Completed** — list which bonus tasks were attempted and their location in the notebooks.
6. **Known Issues & Limitations** — any known bugs, unfinished sections, or environment-specific quirks.
7. **Time Spent** — honest estimate of hours spent on each major component.

**Final Checklist (verify each):**
- [ ] All notebooks run end-to-end without errors (restart kernel and run all cells to confirm).
- [ ] `requirements.txt` is up-to-date (`pip freeze > requirements.txt`).
- [ ] All output files referenced in notebooks exist in the `outputs/` directory.
- [ ] No hardcoded absolute paths (use relative paths throughout).
- [ ] No API keys or secrets committed to the repository.
- [ ] Demo video is ≤ 3 minutes and covers: system running on the video, one technical challenge explained, limitations discussed.

**Output:**
- `README.md`
- Verified directory structure (print the tree).
```
