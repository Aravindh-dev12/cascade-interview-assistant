import os
import time


PARAKEET_CTC_EN_US_FUNCTION_ID = "1598d209-5e27-4d3c-8079-4751568b1081"

# Function IDs used by earlier builds or copied examples that are not the intended
# English Parakeet CTC endpoint for this interview-practice runtime.
KNOWN_NON_PARAKEET_ENGLISH_IDS = {
    "bb0837de-8c7b-481f-9ec8-ef5663e9c1fa",  # previous Nemotron streaming default
    "8473f56d-51ef-473c-bb26-efd4f5def2bf",  # Parakeet zh-TW example
}


def install_parakeet_runtime():
    """Make NVIDIA Parakeet CTC English the realtime streaming ASR backend."""
    from engine.stt_worker import STTWorker

    if getattr(STTWorker, "_parakeet_runtime_installed", False):
        return

    original_init = STTWorker.__init__

    def worker_init(worker, audio_recorder, api_key=None):
        original_init(worker, audio_recorder, api_key=api_key)
        language = os.environ.get("NVIDIA_RIVA_LANGUAGE", worker.language_code or "en-US").strip()
        configured = os.environ.get("NVIDIA_RIVA_FUNCTION_ID", "").strip()
        worker.language_code = language or "en-US"

        if worker.language_code.lower().startswith("en") and (
            not configured or configured in KNOWN_NON_PARAKEET_ENGLISH_IDS
        ):
            worker.function_id = PARAKEET_CTC_EN_US_FUNCTION_ID
            if configured and configured != worker.function_id:
                print(
                    f"[stt] Replaced incompatible ASR function-id {configured} with "
                    f"Parakeet CTC English {worker.function_id}."
                )
        elif configured:
            worker.function_id = configured

    def ensure_client(worker):
        if worker._asr_service is None:
            worker._asr_service = worker._create_client()
            print(
                f"[stt] NVIDIA Parakeet streaming ASR client initialized · "
                f"language={worker.language_code} · function-id={worker.function_id}"
            )
        return worker._asr_service

    def run(worker):
        worker.running = True
        worker.status_updated.emit("NVIDIA Parakeet real-time ASR ready")
        while worker.running:
            if not worker.audio_recorder.is_recording:
                if worker._was_recording:
                    worker._close_all_sessions(join_timeout=0.1)
                    worker._was_recording = False
                time.sleep(0.03)
                continue

            worker._was_recording = True
            try:
                worker._ensure_client()
            except Exception as exc:
                worker.error_occurred.emit(f"NVIDIA Parakeet ASR initialization error: {exc}")
                time.sleep(0.5)
                continue

            mic_chunk, system_chunk = worker.audio_recorder.get_next_audio_chunks()
            now = time.monotonic()
            if mic_chunk is not None and len(mic_chunk):
                worker._process_stream(mic_chunk, "Candidate", now)
            if system_chunk is not None and len(system_chunk):
                worker._process_stream(system_chunk, "Interviewer", now)
            time.sleep(0.015)

        worker._close_all_sessions(join_timeout=0.25)
        print("[stt] NVIDIA Parakeet streaming worker stopped.")

    STTWorker.__init__ = worker_init
    STTWorker._ensure_client = ensure_client
    STTWorker.run = run
    STTWorker._parakeet_runtime_installed = True
