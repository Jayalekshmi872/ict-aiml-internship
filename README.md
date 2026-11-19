# Multilingual Key-Moment Extractor

Finds and cuts short video clips containing **Question-Answer**, **Agreement**, and **Disagreement** moments in captioned YouTube videos. The project includes a Streamlit app for interactive use and a Jupyter/Colab notebook for preparing data, fine-tuning a classifier, and processing a set of videos.

The classifier can use a fine-tuned IndicBERT sequence-pair model. If that model is unavailable, the app and pipeline use multilingual cue-phrase rules as a fallback; those results may be less accurate.

## Features

- Reads YouTube captions in the selected language; it does not automatically transcribe videos without captions.
- Supports English, Hindi, Malayalam, Tamil, Urdu, Telugu, Kannada, Bengali, Marathi, Gujarati, and Punjabi.
- Classifies sentence pairs into neutral, question, agreement, or disagreement.
- Groups questions with nearby answers and selects short segments around detected moments.
- Cuts selected segments into individual MP4 files and lets the Streamlit app play or download them individually or as a ZIP.
- Creates a CSV of extracted moments that can be reviewed in the app.
- Provides a notebook workflow for generating review labels, fine-tuning IndicBERT, and processing multiple videos.

## Project files

| File | Purpose |
|---|---|
| `app.py` | Streamlit user interface for analyzing a YouTube link or viewing saved CSV results. |
| `pipeline.py` | Shared transcript, sentence-processing, classification, segment-selection, and clip-export functions. |
| `multilingual_key_moments.ipynb` | Notebook workflow for preparing data, training/evaluating the classifier, analyzing videos, and exporting clips. |
| `video_metadata.xlsx` | Example spreadsheet used by the notebook. |
| `requirements.txt` | Python dependencies for the app and clip-export workflow. |

## Requirements

- Python and `pip` (Python 3.10 or newer is recommended).
- An internet connection to retrieve captions and access YouTube when cutting clips.
- A YouTube video with captions in a supported language for transcript-based analysis.
- A GPU is recommended for fine-tuning the model. The app can run the classifier on CPU, though inference may be slower.

`imageio-ffmpeg` is included in the requirements and is used to provide FFmpeg for clip processing. You can also install FFmpeg separately and add it to your `PATH`.

## Installation

Run these commands from the project directory.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Streamlit app

```bash
python -m streamlit run app.py
```

Open the local URL printed by Streamlit in your browser.

### Analyze a YouTube link

1. In **Cut clips from a YouTube link**, paste a YouTube URL.
2. Select the spoken language.
3. Click **Find moments and cut clips**.
4. Review the clips by moment type, play them, or download individual MP4s or a ZIP.

The sidebar lets you choose the model location, adjust the segment buffer and confidence thresholds, and set the maximum number of clips to cut per moment type.

### Use saved results

In **Saved results (CSV)**, upload `extracted_moments.csv` created by the notebook, or leave the uploader empty to use that file from the project directory. Select a video and moment types, then cut and review the clips.

## Model setup

The notebook fine-tunes `ai4bharat/IndicBERTv2-MLM-only` and saves the classifier and tokenizer to:

```text
models/indicbert_moments/
```

To use a local model, enter that folder in the app's **Fine-tuned model folder / HF repo id** field. The app also searches the project directory for a saved model. Alternatively, enter a Hugging Face model repository ID containing the fine-tuned classifier and tokenizer.

You can set `MOMENT_MODEL` to the local model folder before starting the app:

```powershell
$env:MOMENT_MODEL = "models/indicbert_moments"
python -m streamlit run app.py
```

If the model cannot be loaded, the app displays a warning and uses the rule-based classifier instead. A fine-tuned model is expected to improve classification quality.

## Notebook workflow

Open `multilingual_key_moments.ipynb` in Jupyter or Google Colab and run the cells in order. The notebook includes IPython/Colab commands and downloads the pretrained model during training, so an internet connection is required.

### Prepare the video spreadsheet

The notebook reads `video_metadata.xlsx`. Provide these columns:

| Column | Description |
|---|---|
| `video_id` | Your identifier for the video. |
| `video_title` | A title to show in results. |
| `video_url` | A valid YouTube video URL. |
| `language` | Spoken language, such as `English`, `Hindi`, or `Malayalam`. |

The notebook extracts the YouTube ID from `video_url` and skips rows where it cannot find a valid ID.

### Prepare and train the model

1. The notebook retrieves captions and caches them in `transcripts.json`.
2. It creates `labels_to_review.csv` from rule-generated candidate labels. Review the `label` column before training. Use `neutral`, `question`, `agreement`, or `disagreement`; blank labels are filled with their generated `weak_label`.
3. The notebook splits examples by timeline within each video: the first 70% for training, the next 15% for validation, and the final 15% for testing.
4. It fine-tunes and evaluates the IndicBERT classifier, then saves the model and tokenizer to `models/indicbert_moments/`.
5. Run the inference cells to create `extracted_moments.csv`. Run the optional clip-export cells to create MP4 clips under `clips/`.

Human-reviewed labels are important: training only on generated weak labels is a demonstration path and may produce lower-quality results. Use a GPU for practical training times.

## Outputs and generated files

Depending on which workflow you run, the project creates:

- `transcripts.json` — cached caption fragments and selected caption languages.
- `labels_to_review.csv` — candidate examples and labels for review.
- `models/indicbert_moments/` and `ckpt/` — trained model and training checkpoints.
- `extracted_moments.csv` — detected moments, timestamps, confidence, and transcript text.
- `clips/` — individual exported MP4 clips.
- `ffmpeg_bin/` — FFmpeg binary directory used by the clip pipeline when needed.

Models and generated clips can be large. Transcript caches and reviewed labels may contain content from source videos; avoid committing generated or sensitive data unless you intend to share it.

## How clips are selected

The pipeline uses captions to build sentence-like units, classifies each unit (with the prior sentence as context), then selects Question-Answer, Agreement, and Disagreement moments. By default, segments are 30–60 seconds long and include a configurable buffer around the detected moment. Clip count and confidence thresholds can be adjusted in the app or notebook.

## Limitations

- Videos without accessible captions cannot be analyzed by the current transcript pipeline; automatic speech recognition is not built into the app.
- Captions, including auto-generated captions, may be incomplete or incorrect.
- Captions generally do not identify speakers, so speaker attribution is not provided.
- Classification quality depends on the model and the quality and coverage of reviewed training examples.
- YouTube availability, access restrictions, or download limits may prevent some clips from being cut.
- Review detected moments before relying on them. Follow YouTube's terms and applicable copyright rules when processing or sharing video clips.

## Troubleshooting

### No captions or no moments found

Confirm that the video has captions and that the selected language is appropriate. The app can report no results when captions are unavailable or no moments pass the configured thresholds.

### Model not found

Confirm that the model field points to a local fine-tuned model folder or a valid Hugging Face repository. A local model folder should contain the saved tokenizer and model files. Without a loadable model, the app uses the rule-based fallback.

### Clip export fails

Confirm that `pip install -r requirements.txt` completed successfully, including `yt-dlp[default]` and `imageio-ffmpeg`. Some videos may restrict access. If YouTube requires an authenticated browser session, you can optionally set `YT_COOKIES_BROWSER` to a browser name that has an active session, then restart the app:

```powershell
$env:YT_COOKIES_BROWSER = "chrome"
python -m streamlit run app.py
```

Do not share browser cookies or commit them to the repository.
