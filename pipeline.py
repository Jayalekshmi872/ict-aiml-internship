"""pipeline.py - multilingual key-moment extraction (shared by the notebook and the Streamlit app).

Moments: question (+answer), agreement, disagreement.
Classifier: IndicBERT fine-tuned on (previous sentence, current sentence) pairs.
Fallback when no fine-tuned model is available: multilingual cue-phrase rules.
"""
import re #	To search for patterns in text and extract YouTube video IDs.
import os, shutil, subprocess, zipfile   # NEW: needed for cutting real video clips
import numpy as np
import pandas as pd

LABELS = ['neutral', 'question', 'agreement', 'disagreement']

#this dictionary connects language names with YouTube's language codes.
LANG_CODES = {
    'english': ['en', 'en-US', 'en-GB', 'en-IN'],
    'hindi': ['hi', 'hi-IN'],
    'malayalam': ['ml', 'ml-IN'],
    'tamil': ['ta', 'ta-IN'],
    'urdu': ['ur', 'ur-PK'],
    'telugu': ['te', 'te-IN'],
    'kannada': ['kn', 'kn-IN'],
    'bengali': ['bn', 'bn-BD', 'bn-IN'],
    'marathi': ['mr', 'mr-IN'],
    'gujarati': ['gu', 'gu-IN'],
    'punjabi': ['pa', 'pa-IN'],
}
#configuration
CFG = dict(
    buffer=25,#before and after detect moment
    min_len=30,#min clip duration 30 sec
    max_len=60,#max clip duration 60 sec
    answer_window=20,#The first answer must start within 20 seconds after a question.
    q_thr=0.60, #	Minimum confidence score for accepting a question prediction.
    ad_thr=0.70, #	Minimum confidence score for accepting agreement/disagreement predictions.
    top_k=8,  #	Maximum 8 selected clips per category per video.
    max_tokens=128)#	Maximum number of tokens given to the model for each sentence pair.

# '.', '?', '!', Devanagari danda, Urdu full stop, Arabic question mark
END_PUNCT = '.?!\u0964\u06D4\u061F'
STRIP_CHARS = '.,!?;:"\'()[]\u0964\u06D4\u061F\u060C-\u2013\u2014\u2026'


def lang_key(name):#checks the language name.
    k = str(name).strip().lower()
    return k if k in LANG_CODES else 'english'
    # Converts it into a string.
    # Removes unnecessary spaces.
    # Converts it to lowercase.
    # Checks whether language exists in LANG_CODES.
    # if language is not in the dict it return english


def extract_video_id(url):#extracts the YouTube video ID from a video link
    m = re.search(r'(?:youtu\.be/|v=|/embed/|/shorts/)([A-Za-z0-9_-]{11})', str(url))
    return m.group(1) if m else None


# Transcript (captions only - no video/audio download)
#retrieves available YouTube captions
def fetch_transcript(yt_id, language='english'):
    """Returns (snippets, language_code_used) or (None, None). Works with youtube-transcript-api 0.x and 1.x."""
    from youtube_transcript_api import YouTubeTranscriptApi
    langs = LANG_CODES[lang_key(language)]#selected vedio language
    try:
        api = YouTubeTranscriptApi()#object
        if hasattr(api, 'fetch'):                                   # v1.x
            try:
                fetched = api.fetch(yt_id, languages=langs)
                #Requests captions for the video in the selected language.
            except Exception:                                       # declared language not found -> any track
                fetched = next(iter(api.list(yt_id))).fetch()
                #It tries to retrieve another available transcript track.
            snips = [{'text': s.text, 'start': s.start, 'duration': s.duration} for s in fetched]
            #       caption,start,end
            return snips, getattr(fetched, 'language_code', None)
        return YouTubeTranscriptApi.get_transcript(yt_id, languages=langs), langs[0]   # v0.x
    except Exception as e:
        print(f'  transcript failed for {yt_id}: {type(e).__name__}: {str(e)[:120]}')
        return None, None


# Caption fragments -> sentences
def to_words(snips):
    words = []
    #Creates an empty list to store individual words.
    for i, s in enumerate(snips):
      #Goes through every caption fragment.
        txt = re.sub(r'\[[^\]]*\]|\([^)]*\)', ' ', s['text']).replace('\n', ' ')
        #remove  [Music], (laughs)
        ws = txt.split()
        #Splits the caption into individual words.
        if not ws:
            continue
        start, dur = s['start'], s['duration']
        #handle overlaping caption
        if i + 1 < len(snips):
            dur = min(dur, max(snips[i + 1]['start'] - start, 0.2))
        dur = max(dur, 0.2)
        for k, w in enumerate(ws):
            words.append({'w': w,#word
                          't': start + dur * k / len(ws), #estimted start time
                          'e': start + dur * (k + 1) / len(ws)}) #estimted end time
    return words

def build_sentences(snips, max_words=30, pause=1.2):
  #max word in group 30,pause of more than 1.2 seconds can indicate a new group
  #combines individual words into sentence-like groups
    words = to_words(snips)
    if not words:
        return [], False
    has_punct = sum(w['w'][-1] in END_PUNCT for w in words) / len(words) > 0.01 ##Why?-it can help identify sentence boundaries.
    sents, cur, turn = [], [], False
#for combiing words use :
  # Punctuation.
  # Time gaps between words.
  # Maximum word limits.
  # Speaker-change markers, if available.


    #sents: Stores the completed sentence groups.
    #cur:   Stores the words in the current group.
    #turn:  Tracks whether a speaker-change marker was detected

    def flush():
        #Saves the collected words as a group
        nonlocal cur, turn
        if cur:
            sents.append({'text': ' '.join(x['w'] for x in cur),
                          'start': cur[0]['t'],
                          'end': cur[-1]['e'],
                          'n': len(cur),
                          'turn': turn})
        cur, turn = [], False

    for w in words:
        tok = w['w']
        if tok.startswith('>>'):  # speaker-change marker
            flush()
            turn = True
            tok = tok.lstrip('>').strip()
            if not tok:
                continue
            w = {**w, 'w': tok}
        elif cur:
            gap = w['t'] - cur[-1]['e'] #previous word - current word
            if (cur[-1]['w'][-1] in END_PUNCT or (not has_punct and (gap > pause or len(cur) >= max_words))
                    or len(cur) >= 45):
                flush()
        cur.append(w)
    flush()
    return sents, has_punct


# Rule-based "weak labels"
#English question words.
EN_WH = r'(what|why|how|when|where|who|whom|whose|which)'
EN_AUX = (r"(do|does|did|is|are|was|were|can|could|would|will|should|have|has|had|am|shall|may|might|"
          r"isn't|aren't|don't|doesn't|didn't|won't)")
EN_Q_START = re.compile(rf'^(so |and |but |okay |ok |well )?({EN_WH}|{EN_AUX})\b', re.I)

#Agreement and disagreement phrases
EN_AGREE = re.compile('|'.join([
    r"i agree", r"i totally agree", r"couldn'?t agree more", r"exactly", r"absolutely", r"you'?re (absolutely )?right",
    r"you are right", r"that'?s (so )?(true|right|correct)", r"definitely", r"for sure", r"well said",
    r"fair (point|enough)", r"i think so too", r"same (here|for me)", r"100 ?%", r"makes sense", r"^yes\b",
    r"^yeah\b", r"^true\b", r"i feel the same"]), re.I)
EN_DISAGREE = re.compile('|'.join([
    r"i disagree", r"i don'?t agree", r"i don'?t think so", r"not really", r"that'?s not (true|right|correct|it)",
    r"i wouldn'?t say", r"i'?m not sure about that", r"on the contrary", r"i beg to differ", r"not necessarily",
    r"i don'?t buy", r"that'?s wrong", r"^no\b", r"^nope\b", r"^but\b", r"^actually\b", r"^however\b",
    r"hold on", r"i('d| would) push back", r"i don'?t think"]), re.I)

#Multilingual cue dictionary
LEX = {
    'hindi': {
        'agree': ['बिल्कुल', 'बिलकुल', 'सही कहा', 'ठीक कहा', 'मैं सहमत हूँ', 'मैं सहमत हूं', 'सही है', 'सच है',
                  'हाँ', 'हां', 'जी हाँ', 'जी हां', 'बिल्कुल सही'],
        'disagree': ['सहमत नहीं', 'असहमत', 'नहीं', 'ऐसा नहीं है', 'ऐसा नहीं', 'मुझे नहीं लगता', 'लेकिन', 'पर',
                     'गलत', 'ग़लत'],
        'question': ['क्या', 'कैसे', 'क्यों', 'कब', 'कहाँ', 'कहां', 'कौन', 'कितना', 'कितने', 'किस', 'किसने']},
    'urdu': {
        'agree': ['بالکل', 'جی ہاں', 'جی', 'ٹھیک ہے', 'درست', 'میں متفق ہوں', 'آپ صحیح کہہ رہے ہیں', 'صحیح کہا'],
        'disagree': ['متفق نہیں', 'اختلاف', 'نہیں', 'ایسا نہیں ہے', 'ایسا نہیں', 'مجھے نہیں لگتا', 'لیکن', 'غلط'],
        'question': ['کیا', 'کیسے', 'کیوں', 'کب', 'کہاں', 'کون', 'کتنا', 'کتنے', 'کس']},
    'tamil': {
        'agree': ['ஆமாம்', 'ஆமா', 'கண்டிப்பா', 'கண்டிப்பாக', 'நிச்சயமா', 'நிச்சயமாக', 'உண்மைதான்', 'சரிதான்',
                  'சரியா சொன்னீங்க', 'நான் ஒத்துக்கிறேன்'],
        'disagree': ['இல்லை', 'இல்ல', 'ஆனா', 'ஆனால்', 'தப்பு', 'தவறு', 'நான் ஒத்துக்கல', 'எனக்கு அப்படி தோணல'],
        'question': ['என்ன', 'எப்படி', 'ஏன்', 'எப்போ', 'எப்போது', 'எங்கே', 'எங்க', 'யார்', 'எவ்வளவு']},
    'malayalam': {
        'agree': ['അതെ', 'ശരിയാണ്', 'ശരിയാ', 'സത്യമാണ്', 'തീർച്ചയായും', 'ഞാൻ യോജിക്കുന്നു'],
        'disagree': ['അല്ല', 'ഇല്ല', 'പക്ഷേ', 'പക്ഷെ', 'തെറ്റാണ്', 'ഞാൻ യോജിക്കുന്നില്ല', 'എനിക്ക് തോന്നുന്നില്ല'],
        'question': ['എന്ത്', 'എന്താണ്', 'എങ്ങനെ', 'എന്തുകൊണ്ട്', 'എപ്പോൾ', 'എവിടെ', 'ആര്', 'എത്ര', 'ആണോ']},
}


def _clean_tokens(text, n=None):
    toks = [t.strip(STRIP_CHARS) for t in text.split()]
    toks = [t for t in toks if t]
    return toks[:n] if n else toks
#It cleans text by removing unnecessary punctuation from the beginning and end of each word.

def _first_hit(text, phrases, head=14):#search only within the first 14 words.
    hay_t = _clean_tokens(text, head)
    #call previous fun [_clean_tokens()]
    #t cleans the sentence and takes the first 14 words.
    hay = ' ' + ' '.join(hay_t) + ' '
    #Joins the cleaned words into one sentence, adding a space at the beginning and end.
    best = None
    for p in phrases:
        idx = hay.find(' ' + p + ' ')
        if idx >= 0:
            pos = hay[:idx + 1].count(' ')
            best = pos if best is None else min(best, pos)
    return best


def weak_label(cur, lang, has_punct, n_words=None):
  #It checks the language, punctuation, question patterns and cue phrases.
  #If it cannot identify any of the special categories, it returns neutral
    """Rule-based label for a sentence: neutral / question / agreement / disagreement."""
    key = lang_key(lang)
    n = n_words if n_words is not None else len(cur.split())
    if '?' in cur or '\u061F' in cur:
        return 'question'
    if key == 'english':
        if not has_punct and EN_Q_START.match(cur.strip()) and 3 <= n <= 28:
            return 'question'
        head = ' '.join(cur.split()[:14])
        a, d = EN_AGREE.search(head), EN_DISAGREE.search(head)
        if a and (not d or a.start() < d.start()):
            return 'agreement'
        if d:
            return 'disagreement'
        return 'neutral'
    lex = LEX.get(key)
    if lex is None:
        return 'neutral'
    toks = set(_clean_tokens(cur))
    if not has_punct and 3 <= n <= 25 and any(q in toks for q in lex['question']):
        return 'question'
    a, d = _first_hit(cur, lex['agree']), _first_hit(cur, lex['disagree'])
    if d is not None and (a is None or d <= a):        # tie -> disagreement ("सहमत नहीं" contains the agree stem)
        return 'disagreement'
    if a is not None:
        return 'agreement'
    return 'neutral'


# Fine-tuned IndicBERT classifier
class MomentClassifier:
  #creates a class responsible for loading the fine-tuned model and using it to classify conversations.
    """Loads a fine-tuned sequence-pair classifier (local folder or Hugging Face repo id)."""

    def __init__(self, path):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        #Loading the tokenizer
        #text-->numbers
        self.model = AutoModelForSequenceClassification.from_pretrained(path).eval()
        #loads the already fine-tuned classification model from the given path.
        #path can point to a locally saved model folder or a Hugging Face model repository.
        #.eval() puts the model into evaluation mode for prediction.
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'#gpu available
        self.model.to(self.dev)#move to selected device
        id2 = self.model.config.id2label
        self.labels = [id2[i] for i in range(len(id2))]

    def predict(self, pairs, bs=32):#batch size = 32.
      #function sends sentence pairs to the model and gets predictions.
        out = []
        with self.torch.no_grad():
            for i in range(0, len(pairs), bs):
                chunk = pairs[i:i + bs]
                #tokenizes the previous and current sentences as a pair.
                enc = self.tok([p for p, _ in chunk],
                 [c for _, c in chunk],
                               truncation=True,#Cuts text if it is too long.
                               max_length=CFG['max_tokens'], #	Maximum token length is 128.
                               padding=True,#	Makes the inputs in a batch the same length.
                               return_tensors='pt').to(self.dev)#Returns PyTorch tensors.
                out.append(self.torch.softmax(self.model(**enc).logits, -1).cpu().numpy())
                #Gets the model's raw classification scores.
                #Converts those scores into normalized confidence-like scores across the four labels.
        return np.vstack(out) if out else np.zeros((0, len(self.labels)))
#simply:This function gives two sentences to the fine-tuned IndicBERT model and gets its classification result.

def classify_sentences(sents, has_punct, lang, clf=None):
  #decides which classification method to use.
    """Returns (labels, confidences) per sentence. Uses the fine-tuned model if given, else the rules."""
    if clf is not None:
      #Fine-tuned model is available-->uses IndicBERT to classify the sentence pairs.
        pairs = [(sents[i - 1]['text'] if i else '', s['text']) for i, s in enumerate(sents)]
        P = clf.predict(pairs)
        idx = P.argmax(1)
        return [clf.labels[k] for k in idx], [float(P[i, k]) for i, k in enumerate(idx)]
        #Fine-tuned model is not available-->uses the rule-based weak-label function instead.
    labs = [weak_label(s['text'], lang, has_punct, s['n']) for s in sents]
    confs = [1.0 if l == 'neutral' else 0.8 if l == 'question' else 0.75 for l in labs]
    return labs, confs


# Moments -> segments [x - y, x + y]
#Finding moments and creating video clips
#project uses the classification results to identify important moments.
def find_moments(sents, labs, confs, cfg=CFG):
    out = []
    for i, s in enumerate(sents):
      #Question–Answer detection
        if labs[i] == 'question' and confs[i] >= cfg['q_thr']:
            ans, j, words = [], i + 1, 0
            while j < len(sents) and len(ans) < 3:
                nx = sents[j]
                if nx['start'] - s['end'] > (cfg['answer_window'] if not ans else 4):
                    break
                if labs[j] == 'question':
                    break
                ans.append(nx)
                words += nx['n']
                j += 1
            if words >= 4:
                conf = min(1.0, confs[i] + 0.1 * ans[0]['turn'])
                out.append({'type': 'Question-Answer', 'm_start': s['start'], 'm_end': ans[-1]['end'],
                            'conf': round(conf, 2),
                            'text': s['text'] + '  ->  ' + ' '.join(a['text'] for a in ans)})
                #If a sentence is classified as a question and its confidence reaches the required threshold, the function searches for:
                  # Whether the answer starts within 20 seconds.
                  # Whether another question appears.
                  # Whether there are enough answer words.
                  # Whether it can collect up to three following sentence groups.
                # system tries to combine the question and its answer into one detected moment.

        #Agreement and disagreement detection
        elif (labs[i] in ('agreement', 'disagreement') and confs[i] >= cfg['ad_thr'] and i > 0
              and s['n'] >= 2 and s['start'] - sents[i - 1]['end'] < 8):
            out.append({'type': labs[i].capitalize(), 'm_start': sents[i - 1]['start'], 'm_end': s['end'],
                        'conf': round(confs[i], 2), 'text': sents[i - 1]['text'] + '  ->  ' + s['text']})
      #If the sentence is classified as agreement or disagreement and its confidence reaches the required threshold, it checks the previous sentence.
      #It then combines the previous and current sentences into a conversation pair.
    return out


def make_segment(m_start, m_end, y, video_len, cfg=CFG):
  #function converts a detected moment into a video clip.
  #if detected moment (500 s) buffer time=25sec so (475:525 )sec
  #also ensures the clip stays within the minimum and maximum duration settings and does not go beyond the video boundaries.
    x = (m_start + m_end) / 2
    # find the middle part (avg) of the detected moment

    half = max(y, (m_end - m_start) / 2 + 2)
    # decides how many seconds should be included on each side of the middle point.
    # To make sure the clip includes enough time before and after the detected moment.

    half = min(max(half, cfg['min_len'] / 2), cfg['max_len'] / 2)
         #Apply minimum (30)and maximum limits(60)

    s, e = x - half, x + half
     # s means clip starting time.
    # e means clip ending time.
    if s < 0:
      #Check if the clip starts before the video
        e, s = e - s, 0.0
    if video_len and e > video_len:
        s, e = max(0.0, s - (e - video_len)), video_len
        #Check if the clip exceeds the video duration
    return s, e, x  # #Return the values(start time,endtime,middle point of the detected moment)


def _overlap(a, b):#Checking whether two clips overlap
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(1e-9, min(a[1] - a[0], b[1] - b[0]))
     #If two clips do not overlap, the result is 0.
    #we dont need duplictaed clips


def select_segments(moments, y, video_len, cfg=CFG):
  # moments: all detected Question–Answer, Agreement and Disagreement moments.
  # y: buffer time.
  # video_len: original video duration.
    segs = [] #list stores all the video clips we calculate.
    for m in moments:
        s, e, x = make_segment(m['m_start'], m['m_end'], y, video_len, cfg)
        #the segs list will now contain the original information plus:
        #- x: middle point.
        #- seg_start: clip starting time.
        #- seg_end: clip ending time.
        #- **m = It copies all the existing information from m and adds the new information.
        segs.append({**m, 'x': x, 'seg_start': s, 'seg_end': e})
    segs.sort(key=lambda r: -r['conf'])
    #sorts clips from highest confidence to lowest confidence.
    #code wants to consider the most confident detections first.
    kept = [] #final selected clips
    for r in segs:
          # Remove overlapping clips of the same type

        if any(k['type'] == r['type'] and _overlap((k['seg_start'], k['seg_end']), (r['seg_start'], r['seg_end'])) > 0.5
               for k in kept):
            continue
        #It only removes overlapping clips when they have the same type.

        if sum(k['type'] == r['type'] for k in kept) >= cfg['top_k']:
            continue  #Limit the number of clips to 8 for each category

        kept.append(r)#Keep the selected clip
    return sorted(kept, key=lambda r: r['seg_start'])
    #Sort clips by starting time (chronological order)
    #Even though the code initially sorted by confidence, the final output is sorted by video time.

def fmt(t):
    t = int(t)
    return f'{t // 3600:02d}:{(t % 3600) // 60:02d}:{t % 60:02d}'
  #This function converts seconds into the familiar video timestamp format(HH:MM:SS)



CLIP_DIR = 'clips'


def ffmpeg_exe():
    """Path to ffmpeg: the one on PATH, otherwise the copy shipped with imageio-ffmpeg.
    FIXED: the bundled binary has an odd name (e.g. ffmpeg-win-x86_64-v7.1.exe) that yt-dlp does not recognise
    ("ffmpeg is not installed"). We copy it to ./ffmpeg_bin/ffmpeg(.exe) and add that folder to PATH."""
    exe = shutil.which('ffmpeg')
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        src = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
    bin_dir = os.path.abspath('ffmpeg_bin')
    dst = os.path.join(bin_dir, 'ffmpeg.exe' if os.name == 'nt' else 'ffmpeg')
    try:
        if not os.path.exists(dst):
            os.makedirs(bin_dir, exist_ok=True)
            shutil.copy2(src, dst)
            os.chmod(dst, 0o755)
        os.environ['PATH'] = bin_dir + os.pathsep + os.environ.get('PATH', '')
        return dst
    except Exception:
        return src


def clip_path(yt_id, start, end, out_dir=CLIP_DIR):
    return os.path.join(out_dir, f'{yt_id}_{int(start)}_{int(end)}.mp4')


def find_clip(yt_id, start, end, out_dir=CLIP_DIR):
    """Returns the path of an already-cut clip, or None."""
    f = clip_path(yt_id, start, end, out_dir)
    return f if os.path.exists(f) and os.path.getsize(f) > 10_000 else None


def clip_duration(path):
    """Length of a video file in seconds (read from ffmpeg's banner), or None."""
    exe = ffmpeg_exe()
    if not exe or not path:
        return None
    r = subprocess.run([exe, '-hide_banner', '-i', path], capture_output=True, text=True)
    m = re.search(r'Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)', r.stderr)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else None


def make_clip(yt_id, start, end, out_dir=CLIP_DIR, max_height=480):
    """Cut seconds [start, end] of a YouTube video into a NEW .mp4 file and return its path.
    Only that section is downloaded. Raises RuntimeError with a readable message if it fails."""
    done = find_clip(yt_id, start, end, out_dir)
    if done:
        return done                                              # already cut earlier -> reuse
    try:
        import yt_dlp
        from yt_dlp.utils import download_range_func
    except ImportError:
        raise RuntimeError('yt-dlp is not installed. Run:  pip install -U yt-dlp')
    ff = ffmpeg_exe()
    if not ff:
        raise RuntimeError('ffmpeg not found. Run:  pip install imageio-ffmpeg   (or install ffmpeg and add it to PATH)')
    os.makedirs(out_dir, exist_ok=True)
    final = clip_path(yt_id, start, end, out_dir)
    stem = final[:-4]
    opts = {
        'format': (f'bv*[height<={max_height}][ext=mp4]+ba[ext=m4a]/b[height<={max_height}][ext=mp4]/'
                   f'bv*[height<={max_height}]+ba/b[height<={max_height}]/b'),
        'download_ranges': download_range_func(None, [(float(start), float(end))]),   # <- the actual cutting
        'force_keyframes_at_cuts': True,                         # frame-accurate start/end
        'merge_output_format': 'mp4',
        'outtmpl': stem + '.%(ext)s',
        'ffmpeg_location': ff,
        'overwrites': True, 'quiet': True, 'no_warnings': True, 'noprogress': True,
    }
    if os.environ.get('YT_COOKIES_BROWSER'):                     # e.g. set YT_COOKIES_BROWSER=chrome if YouTube asks to sign in
        opts['cookiesfrombrowser'] = (os.environ['YT_COOKIES_BROWSER'],)
    last_err = None
    for attempt in range(2):                                     # CHANGED: retry once (network hiccups, ffmpeg code 8)
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([f'https://www.youtube.com/watch?v={yt_id}'])
            last_err = None
            break
        except Exception as e:
            last_err = e
    if last_err is not None:
        raise RuntimeError(f'yt-dlp could not download this section: {str(last_err)[:300]}')
    produced = [f for f in os.listdir(out_dir) if f.startswith(os.path.basename(stem) + '.') and not f.endswith('.part')]
    if not produced:
        raise RuntimeError('No clip file was produced.')
    src = os.path.join(out_dir, produced[0])
    if not src.lower().endswith('.mp4'):                         # webm/mkv -> mp4 so every browser can play it
        subprocess.run([ff, '-y', '-loglevel', 'error', '-i', src, '-c:v', 'libx264', '-c:a', 'aac', final], check=True)
        os.remove(src)
    return final


def _stream_info(yt_id, max_height=480):
    """FIXED: ONE YouTube lookup per video. YouTube often has NO single file with audio+video ("Requested format is
    not available"), so we ask for the best video (<= max_height) + best audio and return BOTH stream URLs."""
    import yt_dlp
    opts = {'format': f'bv*[height<={max_height}]+ba/b[height<={max_height}]/bv*+ba/b',
            'quiet': True, 'no_warnings': True, 'skip_download': True}
    if os.environ.get('YT_COOKIES_BROWSER'):
        opts['cookiesfrombrowser'] = (os.environ['YT_COOKIES_BROWSER'],)
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f'https://www.youtube.com/watch?v={yt_id}', download=False)
    fmts = info.get('requested_formats') or [info]
    return [(f['url'], f.get('http_headers') or info.get('http_headers') or {}) for f in fmts]


def cut_from_stream(inputs, start, end, out_file):
    """FIXED: cut [start, end] seconds from one or two inputs (video stream + audio stream, URLs or local files)
    with ffmpeg and write a NEW mp4 (re-encoded, so the cut is exact)."""
    ff = ffmpeg_exe()
    if not ff:
        raise RuntimeError('ffmpeg not found. Run:  pip install imageio-ffmpeg')
    cmd = [ff, '-y', '-loglevel', 'error']
    for url, headers in inputs:
        if str(url).startswith('http'):
            if headers:
                cmd += ['-headers', ''.join(f'{k}: {v}\r\n' for k, v in headers.items())]
            cmd += ['-reconnect', '1', '-reconnect_streamed', '1', '-reconnect_delay_max', '5']
        cmd += ['-ss', str(start), '-i', url]
    cmd += ['-t', str(end - start)]
    if len(inputs) == 2:
        cmd += ['-map', '0:v:0', '-map', '1:a:0']                 # video from input 0, audio from input 1
    cmd += ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-c:a', 'aac', '-movflags', '+faststart', out_file]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        if os.path.exists(out_file):
            os.remove(out_file)
        raise RuntimeError(r.stderr[-300:])


def export_clips(df, out_dir=CLIP_DIR, progress=None, fast=True, max_per_type=None):
    """CHANGED (much faster): groups moments by video, does ONE YouTube lookup per video, then cuts every
    clip with ffmpeg. Falls back to the slow per-clip download (make_clip) if that fails.
    Resumable: clips that already exist are skipped.
    max_per_type: keep only the N most confident moments per (video, type); the others get clip_file = None."""
    d = df
    if max_per_type:
        d = df.sort_values('confidence', ascending=False).groupby(['yt_id', 'type']).head(max_per_type)
    os.makedirs(out_dir, exist_ok=True)
    paths, done, errs = {}, 0, []
    for yt_id, g in d.groupby('yt_id', sort=False):
        stream = None
        for idx, r in g.iterrows():
            s, e = int(r.seg_start), int(r.seg_end)
            f = find_clip(yt_id, s, e, out_dir)
            if f is None and fast and stream is not False:
                try:
                    if stream is None:
                        stream = _stream_info(yt_id)
                    cut_from_stream(stream, s, e, clip_path(yt_id, s, e, out_dir))
                    f = find_clip(yt_id, s, e, out_dir)
                except Exception as err:
                    print(f'  fast cut failed for {yt_id} ({str(err)[:120]}) -> slow fallback')
                    errs.append(str(err)[:200])
                    if stream is None:
                        stream = False                       # lookup itself failed: don't retry for this video
            if f is None:
                try:
                    f = make_clip(yt_id, s, e, out_dir)
                except Exception as err:
                    print(f'  clip {idx} failed: {str(err)[:150]}')
                    errs.append(str(err)[:200])
            paths[idx] = f
            done += 1
            if progress:
                progress(done, len(d))
        print(f'{yt_id}: {sum(1 for i in g.index if paths.get(i))}/{len(g)} clips ready')
    out = df.copy()
    out['clip_file'] = pd.Series([paths.get(i) for i in df.index], index=df.index, dtype=object)   # None when not cut
    out.attrs['errors'] = errs                                   # NEW: so the app can show WHY a clip failed
    return out


def zip_clips(paths, zip_path='clips.zip'):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            if isinstance(p, str) and os.path.exists(p):
                z.write(p, os.path.basename(p))
    return zip_path



def analyse(yt_id, language='english', title='', video_id='', clf=None, cfg=CFG, snips=None):
  #connects the earlier functions.
    """Full pipeline for one video. Returns (DataFrame of segments, info dict)."""
    lang_used = None
    if snips is None:
        snips, lang_used = fetch_transcript(yt_id, language)
        # If transcript data has not already been provided, it retrieves captions from YouTube.
    if not snips:
        return pd.DataFrame(), {'error': 'no transcript'}
        # Check whether the transcript exists
        # If there is no transcript, it returns an empty table and an error message.
    sents, has_punct = build_sentences(snips)
    # Converts caption fragments into sentence-like groups.
    video_len = snips[-1]['start'] + snips[-1]['duration']
        #approximate total video duration using the final transcript fragment.
    labs, confs = classify_sentences(sents, has_punct, language, clf)
    #Uses IndicBERT if a fine-tuned classifier is supplied. Otherwise, it uses the rules.
    segs = select_segments(find_moments(sents, labs, confs, cfg),
                           cfg['buffer'],
                           video_len,
                           cfg)
    #Detects Question–Answer, Agreement and Disagreement moments and creates clip timings.
    rows = []
    for r in segs:
        in_seg = ' '.join(s['text'] for s in sents if s['start'] >= r['seg_start'] and s['end'] <= r['seg_end'])
        #Prepare the final results
        rows.append({'video_id': video_id,
                     'title': title,
                     'yt_id': yt_id,
                     'language': language,
                     'type': r['type'],
                     'moment_time': fmt(r['x']),
                     'seg_start': round(r['seg_start'], 1),
                     'seg_end': round(r['seg_end'], 1),
                     'duration_s': round(r['seg_end'] - r['seg_start'], 1),
                     'confidence': r['conf'],
                     'key_exchange': r['text'][:400],
                     'segment_transcript': in_seg})
    info = {'fragments': len(snips), 'sentences': len(sents), 'punctuated': has_punct, 'lang_used': lang_used,
            'video_len': video_len, 'label_counts': pd.Series(labs).value_counts().to_dict()}
    return pd.DataFrame(rows), info
#Returns the results as a Pandas DataFrame and additional information about the video.


# Training-data helper (weak labels -> sheet for human review)
#Preparing training data for fine-tuning
def build_training_candidates(stores, neutral_ratio=1.5, seed=0):
    """stores: list of dicts {video_id, language, sents, has_punct}. Returns a DataFrame to review."""
    #prepares a dataset that can review and correct before training IndicBERT.
    rows = []
    for st in stores:#Collect sentences
        S = st['sents']
        #Goes through the processed videos and their sentence-like groups.
        for i, s in enumerate(S):
            rows.append({'video_id': st['video_id'],
                         'language': st['language'],
                         'start': round(s['start'], 1),
                         'prev': S[i - 1]['text'] if i else '',
                         'cur': s['text'],
                         'weak_label': weak_label(s['text'], st['language'], st['has_punct'], s['n'])})
# Creates a table containing:
# - Video ID.
# - Language.
# - Timestamp.
# - Previous sentence.
# - Current sentence.
# - Automatically generated label.
    df = pd.DataFrame(rows)#df
    pos = df[df.weak_label != 'neutral']#contains question, agreement and disagreement
    neu = df[df.weak_label == 'neutral']#contains neutral
    neu = neu.sample(min(len(neu), int(len(pos) * neutral_ratio) + 50), random_state=seed)#reduces the number of neutral examples. (bias)
    out = pd.concat([pos, neu]).sort_values(['video_id', 'start']).reset_index(drop=True)#Combines the non-neutral and selected neutral examples.
    out['label'] = ''          # <- fill / correct this column by hand (leave blank to accept weak_label)
    return out
