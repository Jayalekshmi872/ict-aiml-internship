"""Streamlit app - multilingual key-moment CLIP CUTTER.

Paste a YouTube link -> the app finds every Question-Answer, Agreement and Disagreement moment and CUTS each one
into its own short video file (30-60 s). Only those cut clips are shown - the full video is never played.
Run:  streamlit run app.py
"""
import io
import os
import zipfile

import pandas as pd
import streamlit as st

import pipeline as pl

st.set_page_config(page_title='Key-Moment Clip Cutter', page_icon='✂️', layout='wide')

TYPES = ['Question-Answer', 'Agreement', 'Disagreement']
ICON = {'Question-Answer': '❓', 'Agreement': '🤝', 'Disagreement': '⚡'}
LANGS = ['English', 'Hindi', 'Malayalam', 'Tamil', 'Urdu', 'Telugu', 'Kannada', 'Bengali', 'Marathi', 'Gujarati', 'Punjabi']


# ------------------------------------------------------------------ helpers
def find_model_dir():
    """Finds the fine-tuned model folder (contains config.json + model.safetensors), also when nested."""
    env = os.environ.get('MOMENT_MODEL')
    if env:
        return env
    for root, dirs, files in os.walk('.'):
        if root.count(os.sep) > 4:
            dirs[:] = []
            continue
        if 'config.json' in files and 'model.safetensors' in files:
            return root
    return 'models/indicbert_moments'


@st.cache_resource(show_spinner='Loading fine-tuned IndicBERT model...')
def load_clf(path):
    try:
        return pl.MomentClassifier(path)
    except Exception as e:                                   # model folder missing / not downloaded
        return str(e)


def clip_file_for(row):
    """Path of the CUT clip for this moment, or None if it has not been cut."""
    f = row.get('clip_file') if hasattr(row, 'get') else None
    if isinstance(f, str) and os.path.exists(f):
        return f
    return pl.find_clip(row['yt_id'], int(row['seg_start']), int(row['seg_end']))


def download_name(row):
    return f"{row['type']}_{str(row['moment_time']).replace(':', '-')}.mp4"


def show_clip(row, key):
    """Plays the cut clip (a separate .mp4 file). If it does not exist, nothing from the full video is shown."""
    clip = clip_file_for(row)
    if clip is None:
        st.warning('This clip could not be cut, so nothing is shown (the full video is never played).')
        return
    st.video(clip)
    dur = pl.clip_duration(clip)
    st.caption(f"{ICON.get(row['type'], '')} {row['type']} · confidence {float(row['confidence']):.2f} · "
               f"taken from {row['moment_time']} of the original video"
               + (f' · clip length {dur:.0f} s' if dur else ''))
    st.write(row['key_exchange'])
    with st.expander('Transcript of this clip'):
        st.write(row['segment_transcript'])
    with open(clip, 'rb') as fh:
        st.download_button('⬇️ Download this clip', fh.read(), download_name(row), 'video/mp4', key=f'dl_{key}')


def show_grouped(df, key_prefix):
    """Tabs per moment type, two clips per row."""
    counts = {t: int((df.type == t).sum()) for t in TYPES}
    tabs = st.tabs([f"{ICON[t]} {t} ({counts[t]})" for t in TYPES])
    for tab, t in zip(tabs, TYPES):
        with tab:
            sub = df[df.type == t].sort_values('confidence', ascending=False).reset_index(drop=True)
            if sub.empty:
                st.info(f'No {t} moments found in this video.')
                continue
            for i in range(0, len(sub), 2):
                cols = st.columns(2)
                for c, j in zip(cols, (i, i + 1)):
                    if j < len(sub):
                        with c:
                            st.markdown(f'**{t} #{j + 1}**')
                            show_clip(sub.iloc[j], f'{key_prefix}_{t}_{j}')


def make_zip(df):
    """ZIP with one folder per moment type."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for _, r in df.iterrows():
            f = clip_file_for(r)
            if f:
                z.write(f, f"{r['type']}/{download_name(r)}")
    return buf.getvalue()


# ------------------------------------------------------------------ sidebar
st.sidebar.title('✂️ Key-Moment Clip Cutter')
model_dir = st.sidebar.text_input('Fine-tuned model folder / HF repo id', find_model_dir())
clf = load_clf(model_dir)
if isinstance(clf, str):
    st.sidebar.warning('Fine-tuned model not found -> using rule-based fallback.\n\n' + clf[:150])
    clf = None
else:
    st.sidebar.success('Fine-tuned IndicBERT loaded')

cfg = dict(pl.CFG)
cfg['buffer'] = st.sidebar.slider('Buffer y (seconds before/after)', 15, 30, 25,
                                  help='Clip = [x - y, x + y] around the moment, kept within 30-60 s')
cfg['ad_thr'] = st.sidebar.slider('Agree/Disagree confidence threshold', 0.5, 0.95, 0.70, 0.05)
cfg['q_thr'] = st.sidebar.slider('Question confidence threshold', 0.5, 0.95, 0.60, 0.05)
max_per_type = st.sidebar.slider('Max clips to cut per type', 1, 8, 3,
                                 help='Cutting takes time, so only the most confident moments of each type are cut.')

tab1, tab2, tab3 = st.tabs(['🔗 Cut clips from a YouTube link', '📂 Saved results (CSV)', 'ℹ️ About'])

# ------------------------------------------------------------------ tab 1: link -> clips
with tab1:
    st.write('Paste a YouTube link (podcast / interview / debate **with captions**) and choose its spoken language. '
             'You get a separate **cut video clip** for each Question-Answer, Agreement and Disagreement moment.')
    c1, c2 = st.columns([3, 1])
    url = c1.text_input('YouTube URL')
    lang = c2.selectbox('Spoken language', LANGS)
    if st.button('✂️ Find moments and cut clips', type='primary'):
        yt_id = pl.extract_video_id(url)
        if not yt_id:
            st.error('Could not read a video id from that URL.')
        else:
            with st.spinner('Reading captions and finding moments...'):
                res, info = pl.analyse(yt_id, lang, title=url, video_id=yt_id, clf=clf, cfg=cfg)
            if res.empty:
                st.error('No captions available, or no moments found. Try another video.')
            else:
                if info.get('lang_used') and not str(info['lang_used']).lower().startswith(
                        pl.LANG_CODES[pl.lang_key(lang)][0]):
                    st.warning(f"Captions were in '{info['lang_used']}', not {lang}. Results may be unreliable.")
                st.caption(f"{info['sentences']} sentences analysed · {len(res)} moments found · "
                           f"cutting the top {max_per_type} per type")
                bar = st.progress(0.0, text='Cutting clips...')
                cut = pl.export_clips(res, max_per_type=max_per_type,
                                      progress=lambda i, n: bar.progress(i / n, text=f'Cutting clip {i} of {n}...'))
                bar.empty()
                st.session_state['cut'] = {'df': cut, 'yt_id': yt_id, 'errors': cut.attrs.get('errors', [])}

    state = st.session_state.get('cut')
    if state:
        cut = state['df']
        ok = cut[cut.clip_file.notna()]
        if ok.empty:
            st.error('No clip could be cut. Most likely YouTube blocked the download or yt-dlp/ffmpeg is missing.')
            for e in state['errors'][:3]:
                st.code(e)
            st.caption('Try: pip install -U yt-dlp imageio-ffmpeg   ·   if YouTube asks to sign in, set the '
                       'environment variable YT_COOKIES_BROWSER=chrome and restart the app.')
        else:
            m1, m2, m3, m4 = st.columns(4)
            for col, t in zip((m1, m2, m3), TYPES):
                col.metric(f'{ICON[t]} {t}', int((ok.type == t).sum()))
            m4.metric('Clips cut', len(ok))
            if state['errors']:
                st.warning(f"{len(state['errors'])} clip(s) failed and are not shown: {state['errors'][0][:150]}")
            st.download_button('⬇️ Download all clips (ZIP, one folder per type)', make_zip(ok),
                               f"clips_{state['yt_id']}.zip", 'application/zip')
            show_grouped(ok, state['yt_id'])

# ------------------------------------------------------------------ tab 2: CSV from the notebook
with tab2:
    up = st.file_uploader('Upload extracted_moments.csv (or leave empty to use the one in this folder)', type='csv')
    df = pd.read_csv(up) if up is not None else (pd.read_csv('extracted_moments.csv')
                                                 if os.path.exists('extracted_moments.csv') else None)
    if df is None or df.empty:
        st.info('No saved results found. Use the first tab, or run the notebook to create extracted_moments.csv.')
    else:
        if 'language' not in df.columns:
            df['language'] = 'English'
        f1, f2 = st.columns(2)
        titles = sorted(df.title.astype(str).unique())
        vsel = f1.selectbox('Video', titles)
        tsel = f2.multiselect('Moment type', TYPES, default=TYPES)
        view = df[(df.title.astype(str) == vsel) & df.type.isin(tsel)].sort_values('confidence', ascending=False)
        view = view.groupby('type').head(max_per_type).reset_index(drop=True)
        st.write(f'**{len(view)}** moments (top {max_per_type} per type)')
        if st.button('✂️ Cut these clips'):
            bar = st.progress(0.0, text='Cutting clips...')
            cut = pl.export_clips(view, max_per_type=max_per_type,
                                  progress=lambda i, n: bar.progress(i / n, text=f'Cutting clip {i} of {n}...'))
            bar.empty()
            st.session_state['csv_cut'] = (vsel, cut)
            if cut.attrs.get('errors'):
                st.warning(cut.attrs['errors'][0][:200])
        saved = st.session_state.get('csv_cut')
        if saved and saved[0] == vsel:
            ok = saved[1][saved[1].clip_file.notna()]
            if len(ok):
                show_grouped(ok, f'csv_{vsel}')
        else:
            st.caption('Click "Cut these clips" to create and play the clips for this video.')

# ------------------------------------------------------------------ tab 3
with tab3:
    st.markdown("""
**What it does** Paste a YouTube link. The app reads the captions (no full-video download), finds
Question-Answer, Agreement and Disagreement moments with a fine-tuned **IndicBERT** model, and **cuts each moment into its
own 30-60 s video file** (`[x - y, x + y]` around the moment). Only the cut clips are played or downloaded.

**Languages** English, Hindi, Malayalam, Tamil, Urdu (IndicBERT v2 covers 20+ Indian languages).

**Needs** `yt-dlp` and `ffmpeg` (installed through `pip install -r requirements.txt`).

**Limits** captions have no speaker labels, auto-captions can be wrong, and accuracy depends on the quality and
size of the labelled training data. Cutting takes time, so only the most confident moments per type are cut.
""")
