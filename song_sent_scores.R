# ==========================================
# --- Load Packages ---
# ==========================================
library(reticulate)
library(tictoc)

# ===========================================
# --- SETUP PYTHON CACHING INFRASTRUCTURE ---
# ===========================================
# Define the Python code string 

python_cache_setup_code <- "
import torch, librosa, numpy as np, warnings, gc, time, os
from transformers import AutoProcessor, AutoModel, pipeline

# Suppress warnings
warnings.filterwarnings('ignore', message='.*Maximum duration.*')
warnings.filterwarnings('ignore', message='.*Using PipelineChunkIterator.*')
# --- CORREÇÃO AQUI: Duplicar as contrabarras para R ---
warnings.filterwarnings('ignore', message=r'.*`np\\\\.(bool|int|float)_` was removed.*', category=FutureWarning)
warnings.filterwarnings('ignore', message=r'.*`np\\\\.object_` was removed.*', category=FutureWarning)
warnings.filterwarnings('ignore', message=r'.*copy\\\\(\\\\) is deprecated.*', category=FutureWarning)
# Adicione outras supressões de warning se necessário, escapando barras

# --- Device Setup ---
_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'[Py Setup] Python using device: {_device}')

# --- Global Dictionaries for Caching Models ---
_loaded_clap_models = {}
_loaded_asr_pipelines = {}
_loaded_nli_pipelines = {}

# --- Helper Functions ---
def _clear_gpu_cache():
    gc.collect()
    if _device.type == 'cuda':
        torch.cuda.empty_cache()

def _release_object(obj):
    obj = None

# --- _get_audio_segment (syntax should be fine now) ---
def _get_audio_segment(audio_fpath, target_sr, offset=None, duration=None, verbose=False):
    actual_duration = 0.0; audio_array = None
    try:
        audio_array, sr_ = librosa.load(audio_fpath, sr=target_sr, mono=True, offset=offset, duration=duration, res_type='kaiser_fast')
        # This check is now expected to parse correctly
        if not isinstance(audio_array, np.ndarray) or audio_array.size == 0 or np.max(np.abs(audio_array)) < 1e-6 :
             if verbose: print(f'  [Py Audio] Warning: Loaded audio array empty/silent.')
             return None, 0.0
        actual_duration = librosa.get_duration(y=audio_array, sr=sr_)
        return audio_array.astype(np.float32), actual_duration
    except Exception as e:
        print(f'  [Py Audio] Error loading {os.path.basename(audio_fpath)}: {e}')
        return None, 0.0

# --- Model Loading/Caching Functions ---
# (_get_clap_model, _get_asr_pipeline, _get_nli_pipeline functions remain the same)
def _get_clap_model(model_ident, verbose=False):
    if model_ident in _loaded_clap_models:
        if verbose: print(f'  [Py Cache] Reusing CLAP: {model_ident}')
        return _loaded_clap_models[model_ident]
    else:
        if verbose: print(f'  [Py Cache] Loading CLAP: {model_ident}...')
        st = time.time()
        try:
            processor = AutoProcessor.from_pretrained(model_ident)
            model = AutoModel.from_pretrained(model_ident).to(_device)
            model.eval()
            _loaded_clap_models[model_ident] = {'processor': processor, 'model': model}
            if verbose: print(f'  [Py Cache] CLAP loaded to {_device} in {time.time() - st:.2f}s.')
            return _loaded_clap_models[model_ident]
        except Exception as e:
            print(f'  [Py Cache] Error loading CLAP {model_ident}: {e}'); return None

def _get_asr_pipeline(model_ident, verbose=False):
    if model_ident in _loaded_asr_pipelines:
        if verbose: print(f'  [Py Cache] Reusing ASR: {model_ident}')
        return _loaded_asr_pipelines[model_ident]
    else:
        if verbose: print(f'  [Py Cache] Loading ASR: {model_ident}...')
        st = time.time()
        try:
            asr_pipe = pipeline('automatic-speech-recognition', model=model_ident, device=_device,
                                chunk_length_s=30, stride_length_s=5)
            _loaded_asr_pipelines[model_ident] = asr_pipe
            if verbose: print(f'  [Py Cache] ASR loaded to {_device} in {time.time() - st:.2f}s.')
            return asr_pipe
        except Exception as e:
            print(f'  [Py Cache] Error loading ASR {model_ident}: {e}'); return None

def _get_nli_pipeline(model_ident, verbose=False):
    if model_ident in _loaded_nli_pipelines:
        if verbose: print(f'  [Py Cache] Reusing NLI: {model_ident}')
        return _loaded_nli_pipelines[model_ident]
    else:
        if verbose: print(f'  [Py Cache] Loading NLI: {model_ident}...')
        st = time.time()
        try:
            nli_pipe = pipeline('zero-shot-classification', model=model_ident, device=_device)
            _loaded_nli_pipelines[model_ident] = nli_pipe
            if verbose: print(f'  [Py Cache] NLI loaded to {_device} in {time.time() - st:.2f}s.')
            return nli_pipe
        except Exception as e:
            print(f'  [Py Cache] Error loading NLI {model_ident}: {e}'); return None

# --- Core Analysis Helpers (Using Cache) ---
# (get_clap_scores_pair_py, transcribe_audio_py, get_nli_scores_pair_py functions remain the same)
def get_clap_scores_pair_py(audio_fpath, class_pair, model_ident, offset=None, duration=None, verbose_py=False):
    clap_components = _get_clap_model(model_ident, verbose_py)
    if clap_components is None: return {'error': f'CLAP model {model_ident} failed load/cache.', 'scores': [], 'duration': 0.0}
    processor, model = clap_components['processor'], clap_components['model']
    audio_array, actual_dur = _get_audio_segment(audio_fpath, getattr(processor.feature_extractor, 'sampling_rate', 48000), offset, duration, verbose_py)
    if audio_array is None: return {'error': 'CLAP audio load failed/empty.', 'scores': [], 'duration': actual_dur}
    error_msg, scores = None, []
    try:
        if verbose_py: print(f'  [Py CLAP] Inferring: {class_pair}')
        inputs_dict = processor(text=list(class_pair), audios=[audio_array], return_tensors='pt', padding=True, sampling_rate=getattr(processor.feature_extractor, 'sampling_rate', 48000))
        inputs_dict = {k: v.to(_device) for k, v in inputs_dict.items()}
        with torch.no_grad(): outputs = model(**inputs_dict); probs = torch.softmax(outputs.logits_per_audio[0], dim=0).cpu().numpy(); scores = probs.tolist()
    except Exception as e: error_msg = f'CLAP inference failed: {e}'; scores = []
    finally: _release_object(inputs_dict); _release_object(outputs); _release_object(audio_array); _clear_gpu_cache()
    return {'error': error_msg, 'scores': scores, 'duration': actual_dur}

def transcribe_audio_py(audio_fpath, model_ident, offset=None, duration=None, language=None, verbose_py=False):
    asr_pipe = _get_asr_pipeline(model_ident, verbose_py)
    if asr_pipe is None: return {'error': f'ASR pipeline {model_ident} failed load/cache.', 'text': None, 'duration': 0.0}
    target_sr = getattr(asr_pipe.feature_extractor, 'sampling_rate', 16000)
    audio_array, actual_dur = _get_audio_segment(audio_fpath, target_sr, offset, duration, verbose_py)
    if audio_array is None: return {'error': 'ASR audio load failed/empty.', 'text': None, 'duration': actual_dur}
    err, txt = None, None
    try:
        if verbose_py: print(f'  [Py ASR] Transcribing...')
        generate_kwargs = {'return_timestamps': False}
        if language: generate_kwargs['language'] = language
        with torch.no_grad(): transcription_result = asr_pipe(audio_array.copy(), generate_kwargs=generate_kwargs)
        txt = transcription_result['text'].strip() if transcription_result and 'text' in transcription_result else ''
        if not txt and verbose_py: print('  [Py ASR] Warning: Empty text result.')
    except Exception as e: err = f'ASR transcription failed: {e}'; txt = None
    finally: _release_object(audio_array); _release_object(transcription_result if 'transcription_result' in locals() else None); _clear_gpu_cache()
    return {'error': err, 'text': txt, 'duration': actual_dur}

def get_nli_scores_pair_py(text_input, class_pair, model_ident, verbose_py=False):
    nli_pipe = _get_nli_pipeline(model_ident, verbose_py)
    if nli_pipe is None: return {'error': f'NLI pipeline {model_ident} failed load/cache.', 'scores': []}
    err, scores = None, []
    try:
        if not text_input or not isinstance(text_input, str) or len(text_input.strip()) < 3:
             err = 'NLI input invalid/short.'; return {'error': err, 'scores': []}
        if verbose_py: print(f'  [Py NLI] Inferring: {class_pair}')
        with torch.no_grad(): nli_output = nli_pipe(text_input, candidate_labels=list(class_pair), multi_label=False)
        score_dict = {l: s for l, s in zip(nli_output['labels'], nli_output['scores'])}
        scores = [float(score_dict.get(l, 0.0)) for l in class_pair]
    except Exception as e: err = f'NLI inference failed: {e}'; scores = []
    finally: _release_object(nli_output if 'nli_output' in locals() else None); _clear_gpu_cache()
    return {'error': err, 'scores': scores}


# --- Function to clear caches from R ---
def clear_py_model_caches():
    global _loaded_clap_models, _loaded_asr_pipelines, _loaded_nli_pipelines
    count = len(_loaded_clap_models) + len(_loaded_asr_pipelines) + len(_loaded_nli_pipelines)
    _loaded_clap_models.clear()
    _loaded_asr_pipelines.clear()
    _loaded_nli_pipelines.clear()
    _clear_gpu_cache()
    print(f'[Py Cache] Cleared {count} cached models/pipelines.')
    return count

print('[Py Setup] Caching helper functions defined.')
" # End of python_cache_setup_code string 

# Execute the Python code string to define functions and caches in the Python session
reticulate::py_run_string(python_cache_setup_code)
# Optional: Verify a function exists
# reticulate::py_run_string("print(callable(get_clap_scores_pair_py))")

# ==========================================
# --- Function to Clear Python Caches (Optional) ---
# ==========================================
#' Clears the Python model caches managed by the reticulate session.
#' Call this if you want to force reloading of models without restarting R.
clear_sentiment_model_caches <- function() {
  invisible(reticulate::py$clear_py_model_caches())
}

# ==========================================
# --- MODIFIED song_sent_scores R function (Uses Cached Python functions) ---
# ==========================================
song_sent_scores <- function(audio_path,
                             lyrics = NULL,
                             transcribe_audio = FALSE,
                             start_sec = NULL,
                             end_sec = NULL,
                             clap_model_id = "laion/clap-htsat-unfused",
                             nli_model_id = "joeddav/xlm-roberta-large-xnli",
                             asr_model_id = "openai/whisper-base",
                             asr_language = NULL,
                             verbose = TRUE,
                             verbose_py = FALSE) {
  
  tic("Total song_sent_scores execution (Cached)")
  if (!requireNamespace("reticulate", quietly = TRUE)) stop("Package 'reticulate' required.", call. = FALSE)
  
  # --- CORREÇÃO AQUI: Usar import_main() para checar as funções Python ---
  required_py_funcs <- c("get_clap_scores_pair_py", "transcribe_audio_py", "get_nli_scores_pair_py", "_get_audio_segment", "clear_py_model_caches")
  main_module <- reticulate::import_main(convert = FALSE) # Obter o módulo principal
  py_funcs_exist <- all(sapply(required_py_funcs, reticulate::py_has_attr, x = main_module))
  if (!py_funcs_exist) {
    missing_funcs <- required_py_funcs[!sapply(required_py_funcs, reticulate::py_has_attr, x = main_module)]
    stop("Required Python helper functions not found in main module: ", paste(missing_funcs, collapse=", "),
         ".\nDid you successfully run the 'py_run_string(python_cache_setup_code)' command in this R session?", call. = FALSE)
  }
  
  if (!file.exists(audio_path)) stop("Audio file not found: ", audio_path, call. = FALSE)
  if (!is.null(lyrics) && !is.character(lyrics)) stop("'lyrics' must be a character vector.", call. = FALSE)
  if(transcribe_audio && !is.null(lyrics) && verbose) message("Info: 'lyrics' provided, 'transcribe_audio = TRUE' ignored.")
  
  valence_classes <- c("negative valence", "positive valence")
  arousal_classes <- c("low arousal", "high arousal")
  
  # --- Validate Time Segment (same as before) ---
  offset_py <- reticulate::py_none(); duration_py <- reticulate::py_none()
  segment_duration <- NA_real_; segment_description <- "the entire audio file"
  user_defined_segment <- !is.null(start_sec) || !is.null(end_sec)
  if (user_defined_segment) {
    if (is.null(start_sec) || is.null(end_sec)) stop("Both 'start_sec'/'end_sec' needed if one is set.", call.=F)
    if (!is.numeric(start_sec) || start_sec < 0) stop("'start_sec' must be numeric >= 0.", call.=F)
    if (!is.numeric(end_sec) || end_sec <= start_sec) stop("'end_sec' must be > 'start_sec'.", call.=F)
    offset_py <- reticulate::r_to_py(as.numeric(start_sec)); duration_val <- as.numeric(end_sec - start_sec)
    duration_py <- reticulate::r_to_py(duration_val); segment_duration <- duration_val
    segment_description <- sprintf("segment [%.2fs-%.2fs] (%.2fs)", start_sec, end_sec, duration_val)
  } else { start_sec <- 0; end_sec <- NA_real_ }
  if (verbose) message(paste("Analysis requested for", segment_description))
  
  # --- Initialize Results List (same as before) ---
  results <- list( audio_scores=setNames(rep(NA_real_,4), c("neg_valence","pos_valence","low_arousal","high_arousal")), text_scores=NULL, valence_classes=valence_classes, arousal_classes=arousal_classes, transcribed_text=NULL, text_source="none", models_used=list(clap=clap_model_id, nli=NULL, asr=NULL), segment_info=list(start_sec=start_sec, end_sec=end_sec, duration_analyzed_s=segment_duration))
  
  # --- Helper Normalization (same as before) ---
  normalize_scores <- function(scores) { if (is.null(scores)||length(scores)!=2||!is.numeric(scores)||anyNA(scores)) return(c(NA_real_,NA_real_)); scores[scores<0]<-0; total=sum(scores); if (total>1e-9) return(scores/total) else return(c(0.5,0.5)) }
  
  # --- Determine Actual Segment Duration (using the pre-defined Python function) ---
  actual_audio_duration <- NA_real_; duration_check_ok <- FALSE
  if (verbose) message("Determining analysis segment duration...")
  tryCatch({
    duration_info <- reticulate::py$`_get_audio_segment`(audio_path, 16000L, offset_py, duration_py, verbose = FALSE)
    segment_duration_py <- reticulate::py_to_r(duration_info[[2]])
    if(is.numeric(segment_duration_py) && segment_duration_py > 0){
      results$segment_info$duration_analyzed_s <- segment_duration_py
      if(!user_defined_segment){ results$segment_info$end_sec <- start_sec + segment_duration_py }
      if(verbose) message(sprintf("Actual segment duration: %.2f s", segment_duration_py))
      duration_check_ok <- TRUE
    } else { warning("Could not get valid segment duration from Python.", call. = FALSE) }
  }, error = function(e){ warning("R Error getting segment duration from Python: ", e$message, call. = FALSE) })
  if(!duration_check_ok) {
    if(verbose) message("Using requested duration or NA; Python duration check failed or returned 0.")
    results$segment_info$duration_analyzed_s <- ifelse(user_defined_segment, segment_duration, NA_real_)
  }
  
  
  # --- 1. Audio Analysis (CLAP - Calls pre-defined Python function) ---
  tic("CLAP Analysis (Cached)")
  if(verbose) message(paste("Initiating AUDIO analysis (CLAP):", clap_model_id))
  audio_valence_py <- NULL; audio_arousal_py <- NULL
  tryCatch({
    if(verbose) message(" -> Calling Python CLAP Valence/Arousal...")
    audio_valence_py <- reticulate::py$get_clap_scores_pair_py(audio_path, valence_classes, clap_model_id, offset_py, duration_py, verbose_py)
    audio_arousal_py <- reticulate::py$get_clap_scores_pair_py(audio_path, arousal_classes, clap_model_id, offset_py, duration_py, verbose_py)
  }, error = function(e){ warning("R Error calling Python CLAP function: ", e$message, call. = FALSE) })
  
  py_err_val <- audio_valence_py$error
  py_err_aro <- audio_arousal_py$error
  if(is.null(py_err_val) && is.null(py_err_aro)){
    val_scores_r <- reticulate::py_to_r(audio_valence_py$scores)
    aro_scores_r <- reticulate::py_to_r(audio_arousal_py$scores)
    val_n <- normalize_scores(val_scores_r)
    aro_n <- normalize_scores(aro_scores_r)
    results$audio_scores[] <- c(val_n, aro_n)
    if(verbose) message("Audio analysis completed.")
  } else {
    warning("Python CLAP failed: [Val:", ifelse(is.null(py_err_val), "OK", py_err_val),
            "][Aro:", ifelse(is.null(py_err_aro), "OK", py_err_aro), "]", call. = FALSE)
  }
  toc(log = FALSE)
  
  # --- 2. Text Prep / Transcription (Calls pre-defined Python function) ---
  tic("Text Prep/Transcription (Cached)")
  text_to_analyze <- NULL; perform_text_analysis_step <- TRUE; text_source_final <- "none"
  if(!is.null(lyrics)){
    if(verbose) message("Using provided lyrics.")
    text_to_analyze <- paste(lyrics, collapse = "\n")
    text_source_final <- "provided_lyrics"
  } else if (transcribe_audio){
    results$models_used$asr <- asr_model_id
    if(verbose) message(paste("Initiating TRANSCRIPTION (ASR):", asr_model_id))
    asr_result_py <- NULL
    tryCatch({
      asr_result_py <- reticulate::py$transcribe_audio_py(
        audio_path, asr_model_id, offset_py, duration_py,
        if(!is.null(asr_language)) asr_language else reticulate::py_none(),
        verbose_py
      )
    }, error = function(e){ warning("R Error calling Python ASR function: ", e$message, call. = FALSE) })
    
    py_err_asr <- asr_result_py$error
    if(!is.null(asr_result_py) && is.null(py_err_asr)){
      results$transcribed_text <- reticulate::py_to_r(asr_result_py$text)
      if(!is.null(results$transcribed_text) && is.character(results$transcribed_text) && nchar(results$transcribed_text) > 0){
        if(verbose) message(paste("Transcription OK (", nchar(results$transcribed_text), "chars)."))
        text_to_analyze <- results$transcribed_text
        text_source_final <- "transcribed"
      } else {
        if(verbose) message("Transcription returned empty text.")
        text_source_final <- "transcribed_empty"; perform_text_analysis_step <- FALSE
      }
    } else {
      err_msg <- ifelse(is.null(asr_result_py), "ASR call failed in R", as.character(py_err_asr))
      warning("Transcription failed: ", err_msg, call. = FALSE)
      text_source_final <- "transcribed_failed"; perform_text_analysis_step <- FALSE
    }
  } else {
    if(verbose) message("Skipping text analysis (no lyrics provided, transcribe_audio=FALSE).")
    text_source_final <- "none"; perform_text_analysis_step <- FALSE
  }
  results$text_source <- text_source_final
  toc(log = FALSE)
  
  # --- 3. Text Analysis (NLI - Calls pre-defined Python function) ---
  if(perform_text_analysis_step && !is.null(text_to_analyze) && nchar(text_to_analyze) > 0){
    tic("NLI Analysis (Cached)")
    results$models_used$nli <- nli_model_id
    if(verbose) message(paste("Initiating TEXT analysis (NLI):", nli_model_id))
    text_valence_py <- NULL; text_arousal_py <- NULL
    tryCatch({
      if(verbose) message(" -> Calling Python NLI Valence/Arousal...")
      text_valence_py <- reticulate::py$get_nli_scores_pair_py(text_to_analyze, valence_classes, nli_model_id, verbose_py)
      text_arousal_py <- reticulate::py$get_nli_scores_pair_py(text_to_analyze, arousal_classes, nli_model_id, verbose_py)
    }, error = function(e){ warning("R Error calling Python NLI function: ", e$message, call. = FALSE) })
    
    py_err_val <- text_valence_py$error
    py_err_aro <- text_arousal_py$error
    if(is.null(py_err_val) && is.null(py_err_aro)){
      val_scores_r <- reticulate::py_to_r(text_valence_py$scores)
      aro_scores_r <- reticulate::py_to_r(text_arousal_py$scores)
      val_n <- normalize_scores(val_scores_r)
      aro_n <- normalize_scores(aro_scores_r)
      results$text_scores <- setNames(c(val_n, aro_n), c("neg_valence", "pos_valence", "low_arousal", "high_arousal"))
      if(verbose) message("Text analysis completed.")
    } else {
      warning("Python NLI failed: [Val:", ifelse(is.null(py_err_val), "OK", py_err_val),
              "][Aro:", ifelse(is.null(py_err_aro), "OK", py_err_aro), "]", call. = FALSE)
      results$text_scores <- NULL
    }
    toc(log = FALSE)
  } else {
    if(text_source_final != "none" && perform_text_analysis_step) {
      if(verbose) message("Text analysis skipped due to lack of valid text input.")
    }
    results$text_scores <- NULL
  }
  
  # --- Final Return ---
  if (verbose) message("Processing completed.")
  # toc(log = TRUE) # Log total time
  return(results)
}



# ==========================================
# --- Example Usage (Assumes Python Setup was run) ---
# ==========================================
# my_song_path <- "path/to/your/audio.mp3" # ADJUST
#
# if (file.exists(my_song_path)) {
#   # First call (loads models into Python cache)
# results1 <- song_sent_scores(
#   audio_path = my_song,
#   transcribe_audio = TRUE,
#   start_sec = 10, end_sec = 25,
#   asr_language = "pt", 
#   clap_model_id = "laion/clap-htsat-unfused",
#   nli_model_id = "joeddav/xlm-roberta-large-xnli",
#   asr_model_id = "openai/whisper-large-v3",
#   verbose = TRUE, verbose_py = TRUE # Show Python logs for first run
#  )
# ==========================================
