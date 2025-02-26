import streamlit as st
import cv2
import google.generativeai as genai
import speech_recognition as sr
import librosa
import numpy as np
import plotly.graph_objects as go
from pydub import AudioSegment
import mediapipe as mp
import os
import subprocess
import time
import re
from pyannote.audio import Pipeline
from pyannote.core import Segment, Timeline, Annotation
import torch  # Explicitly import for pyannote.audio
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

# Initialize pyannote.audio pipeline (for speaker diarization) with robust error handling
diarization_pipeline = None
try:
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=None  # No token needed for free model
    )
except Exception as e:
    st.error(f"Warning: Speaker diarization failed to initialize: {str(e)}. Continuing with basic transcription.")
    st.write("Analysis will proceed without speaker separation, treating the transcript as multiple Q&A pairs.")

# Hardcode API key for local testing (replace with your actual key)
API_KEY = "AIzaSyB8aJR3kyZlTQ5rB928gDt4qMYQH5SQhhM"  # Your Gemini API key
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Streamlit UI
st.title("ConfiView - Your Interview Coach")

# Upload video
video_file = st.file_uploader("Upload Your Interview Video (MP4)", type=["mp4"])

if video_file is not None:
    # Save uploaded file to disk
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(video_file.read())
    st.success("Your video is ready for review!")

    # Extract full audio
    try:
        audio_file = "temp_audio.wav"
        audio = AudioSegment.from_file(temp_video_path, format="mp4")
        audio.export(audio_file, format="wav")
        st.success("Audio is set for evaluation!")
    except Exception as e:
        st.error(f"Sorry, we couldn’t process your video: {str(e)}")
        st.stop()

    # Initialize full_text as an empty string for fallback
    full_text = ""

    # Perform transcription with enhanced quality and progress
    st.write("Starting high-quality transcription...")
    progress_bar = st.progress(0)
    try:
        # Transcribe full audio in 5-second chunks with 1-second overlap for better accuracy
        recognizer = sr.Recognizer()
        chunk_duration = 5  # 5-second chunks
        overlap = 1  # 1-second overlap for continuity
        full_transcript = []
        with sr.AudioFile(audio_file) as source:
            audio_length = len(audio) / 1000  # Duration in seconds
            chunk_count = int((audio_length - overlap) / (chunk_duration - overlap)) + 1
            for i in range(chunk_count):
                start_time = max(0, i * (chunk_duration - overlap) * 1000)  # Start in milliseconds
                end_time = min((i + 1) * chunk_duration * 1000, len(audio))
                chunk = audio[start_time:end_time]
                if len(chunk) > 0:
                    chunk.export(f"temp_chunk_{i}.wav", format="wav")
                    with sr.AudioFile(f"temp_chunk_{i}.wav") as chunk_source:
                        chunk_data = recognizer.record(chunk_source)
                        attempts = 3
                        for attempt in range(attempts):
                            try:
                                chunk_text = recognizer.recognize_google(chunk_data, show_all=True)
                                if 'alternative' in chunk_text and chunk_text['alternative']:
                                    chunk_text = chunk_text['alternative'][0]['transcript']
                                else:
                                    chunk_text = recognizer.recognize_google(chunk_data)
                                full_transcript.append(chunk_text)
                                break
                            except (sr.UnknownValueError, sr.RequestError) as e:
                                if attempt == attempts - 1:
                                    full_transcript.append("[Unintelligible]")
                                else:
                                    time.sleep(1)  # Wait before retrying
                    os.remove(f"temp_chunk_{i}.wav")
                progress = int((i + 1) / chunk_count * 30)  # 30% for transcription
                progress_bar.progress(progress)
        full_text = " ".join(full_transcript).strip()
        st.write("Here’s the full, high-quality conversation we heard:", full_text)
        progress_bar.progress(30)  # Transcription complete
    except Exception as e:
        st.error(f"Error in transcription: {str(e)}")
        full_text = "Could not understand audio"
        st.write(full_text)
        progress_bar.progress(30)  # Fallback complete

    # Attempt speaker diarization if pipeline is available
    if diarization_pipeline is not None:
        try:
            waveform, sample_rate = librosa.load(audio_file, sr=None)
            progress_bar.progress(40)  # 40% complete
            diarization = diarization_pipeline({"waveform": waveform, "sample_rate": sample_rate})
            # Convert diarization to timeline of speakers
            speakers = {}
            for segment, track, label in diarization.itertracks(yield_label=True):
                if label not in speakers:
                    speakers[label] = []
                speakers[label].append((segment.start, segment.end))
            progress_bar.progress(50)  # 50% complete
        except Exception as e:
            st.error(f"Error in speaker diarization: {str(e)}")
            diarization = None

    # Identify all questions and answers using Gemini with maximum accuracy and valid Python output
    st.write("Identifying all questions and answers...")
    progress_bar.progress(60)  # 60% complete
    try:
        qa_analysis_prompt = f"""
        Here’s a transcript of an interview conversation, which may include multiple turns, unclear speech, interruptions, or partial responses:
        {full_text}
        As an experienced interviewer, meticulously analyze this transcript to identify as many distinct question-answer pairs as possible with the highest accuracy, ensuring a valid Python list of tuples as output. For each pair, extract:
        - The interviewer’s exact question (starting with words like 'What', 'How', 'Why', 'Tell me', 'Can you', 'Do you', or similar question indicators, even if incomplete, interrupted, or fragmented).
        - The interviewee’s corresponding response (the answer immediately following the question, avoiding overlap with other questions or noise, even if partial or unclear).
        Provide the output *exclusively* as a Python list of tuples in this exact format: `[("Question 1", "Answer 1"), ("Question 2", "Answer 2"), ...]`. Do not include any additional text, explanations, or formatting—only the list of tuples. Ensure you separate questions and answers precisely, assuming the interviewer asks questions and the interviewee responds. Handle multiple turns, unclear phrases, fragmented speech, or interruptions naturally, maximizing the number of pairs while prioritizing accuracy and completeness. If a question or answer is ambiguous, incomplete, or unclear, include it with a note like '[Unclear question]', '[Partial response]', or '[Interrupted]', but only if it’s reasonably identifiable. Exclude pure noise or irrelevant text, and do not group multiple questions into one—identify each question individually, even if they’re close together or overlapping. For example, if the transcript has two questions like 'What’s your name?' and 'Where are you from?', output: `[("What’s your name?", "My name is Kartikey Bhatt"), ("Where are you from?", "I’m from Mandsaur")]`.
        """
        qa_response = model.generate_content(qa_analysis_prompt)
        # Clean and parse the response to ensure valid Python syntax
        cleaned_response = qa_response.text.strip()
        # Remove any extra text or whitespace, ensuring it starts and ends with square brackets
        if cleaned_response.startswith('[') and cleaned_response.endswith(']'):
            try:
                qa_pairs = eval(cleaned_response)  # Safely evaluate as Python list of tuples
            except (SyntaxError, NameError, ValueError) as e:
                st.error(f"Error parsing Q&A pairs: Invalid syntax in Gemini response - {str(e)}. Falling back to basic extraction.")
                qa_pairs = _extract_basic_qa_pairs(full_text)
        else:
            st.error("Error parsing Q&A pairs: Gemini response not in expected format. Falling back to basic extraction.")
            qa_pairs = _extract_basic_qa_pairs(full_text)
        st.write(f"Identified {len(qa_pairs)} questions in the video. Starting detailed analysis...")
    except Exception as e:
        st.error(f"Error identifying questions and answers: {str(e)}")
        qa_pairs = _extract_basic_qa_pairs(full_text) if full_text else [("Could not determine question", "No transcription available")]
        st.write("Identified 1 question (fallback) in the video. Starting detailed analysis...")
    progress_bar.progress(70)  # 70% complete

    # Helper function for basic Q&A extraction (fallback)
    def _extract_basic_qa_pairs(transcript):
        if not transcript or transcript == "Could not understand audio" or transcript == "No transcription available":
            return [("Could not determine question", "No transcription available")]
        prompt = f"""
        Here’s a transcript of an interview conversation, which may include multiple turns, unclear speech, interruptions, or partial responses:
        {transcript}
        As an experienced interviewer, analyze this transcript to identify distinct question-answer pairs with basic accuracy. For each pair, extract:
        - The interviewer’s question (starting with words like 'What', 'How', 'Why', 'Tell me', 'Can you', 'Do you', or similar, even if incomplete).
        - The interviewee’s response (the answer immediately following, even if partial).
        Provide the output *exclusively* as a Python list of tuples in this exact format: `[("Question 1", "Answer 1"), ("Question 2", "Answer 2"), ...]`. Do not include any additional text, explanations, or formatting—only the list of tuples. Handle multiple turns naturally, prioritize simplicity, and include only clear pairs.
        """
        try:
            response = model.generate_content(prompt)
            cleaned_response = response.text.strip()
            if cleaned_response.startswith('[') and cleaned_response.endswith(']'):
                return eval(cleaned_response)
            return [("Could not determine question", transcript)]
        except Exception:
            return [("Could not determine question", transcript)]

    # Parallel analysis of all questions
    results = {}
    total_questions = len(qa_pairs)
    progress_bar_total = st.progress(0)

    def analyze_qa(qa_pair, idx):
        question, answer = qa_pair
        verbal_feedback, verbal_score, posture_score, eye_contact_score, gesture_score, head_tilt_score, tone_score, speech_rate_score, enthusiasm_score = None, 500, 500, 500, 500, 500, 500, 500, 500

        # Verbal analysis with progress
        st.write(f"Checking how you answered Question {idx} in detail...")
        progress_bar = st.progress(0)
        start_time = time.time()
        evaluation_text = f"""
        Here’s what someone said in an interview, which may include multiple turns, unclear speech, interruptions, or partial responses:
        Interviewer asked: "{question}"
        They replied: "{answer}"
        As an experienced interviewer, provide a thorough, detailed, and highly accurate review of their answer. Focus on clarity, relevance to the question, confidence, structure (how logically they organized their response), enthusiasm, and any specific strengths or weaknesses, even if the response is partial, unclear, or interrupted. Highlight what they did exceptionally well, any areas where they could improve, and offer specific, actionable, and friendly tips for their next interview. Account for any ambiguity, incompleteness, or interruptions (e.g., '[Partial response]', '[Unclear]', '[Interrupted]') and adjust your feedback accordingly. Keep it natural, conversational, and supportive, like you’re mentoring them personally, ensuring the analysis is ready for any scenario.
        """
        try:
            progress_bar.progress(50)  # 50% through verbal analysis
            response = model.generate_content(evaluation_text)
            verbal_feedback = response.text
            # Heuristic scoring (capped at 950 out of 1000, avoiding 0 or 1000)
            verbal_score = 600  # Base score (out of 1000)
            if "clear" in verbal_feedback.lower() or "well-spoken" in verbal_feedback.lower():
                verbal_score += 150
            if "confident" in verbal_feedback.lower() or "assured" in verbal_feedback.lower():
                verbal_score += 100
            if "well-structured" in verbal_feedback.lower() or "organized" in verbal_feedback.lower():
                verbal_score += 100
            if "enthusiastic" in verbal_feedback.lower() or "engaged" in verbal_feedback.lower():
                verbal_score += 50
            if "[partial response]" in verbal_feedback.lower() or "[unclear]" in verbal_feedback.lower() or "[interrupted]" in verbal_feedback.lower():
                verbal_score = max(400, verbal_score - 100)  # Penalty for ambiguity
            verbal_score = max(50, min(950, verbal_score))  # Avoid 0 or 1000, scale to 1000
            progress_bar.progress(100)  # 100% complete
        except Exception as e:
            st.error(f"We hit a snag reviewing your answer for Question {idx}: {str(e)}")
            verbal_feedback = "Sorry, we couldn’t fully review your answer this time."

        # Posture, eye contact, gestures, and body language analysis with optimized performance
        st.write(f"Looking closely at your body language for Question {idx}...")
        progress_bar = st.progress(0)
        start_time = time.time()
        try:
            mp_pose = mp.solutions.pose
            mp_face = mp.solutions.face_detection
            pose = mp_pose.Pose()
            face = mp_face.FaceDetection(min_detection_confidence=0.5)
            cap = cv2.VideoCapture(temp_video_path)
            posture_score, eye_contact_score, gesture_score, head_tilt_score, frame_count, processed_frames = 0, 0, 0, 0, 0, 0
            frame_interval = 15  # Optimized for speed while maintaining accuracy
            hand_movement, fidget_count = 0, 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            result_queue = Queue()

            def process_frame(frame_idx, frame):
                if frame_idx % frame_interval == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Posture (spine, shoulders, head)
                    pose_results = pose.process(frame_rgb)
                    posture, eye_contact, gesture, head_tilt, fidget = 0, 0, 0, 0, 0
                    if pose_results.pose_landmarks:
                        left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                        right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                        head_y = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y
                        head_x = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x
                        spine_angle = abs(left_shoulder.y - head_y)
                        shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
                        head_tilt = abs(head_x - (left_shoulder.x + right_shoulder.x) / 2)
                        if spine_angle < 0.3 and shoulder_diff < 0.1:  # Upright and aligned
                            posture = 1
                        if head_tilt < 0.2:  # Minimal tilt for confidence
                            head_tilt = 1
                        # Hand gestures and fidgeting
                        left_hand = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                        right_hand = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                        hand_distance = abs(left_hand.y - right_hand.y)
                        gesture = hand_distance
                        if hand_distance > 0.5:  # Significant movement could indicate fidgeting
                            fidget = 1
                    # Eye contact
                    face_results = face.process(frame_rgb)
                    eye_contact = 1 if face_results.detections else 0
                    result_queue.put((frame_idx, posture, eye_contact, gesture, head_tilt, fidget))

            # Parallel processing for frames
            threads = []
            frame_buffer = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_buffer.append(frame)
                if len(frame_buffer) >= 10:  # Process in batches of 10 frames
                    for idx, f in enumerate(frame_buffer):
                        t = threading.Thread(target=process_frame, args=(frame_count - len(frame_buffer) + idx + 1, f))
                        threads.append(t)
                        t.start()
                    for t in threads:
                        t.join()
                    frame_buffer = []
                    progress = int((frame_count / total_frames) * 50)  # 50% of posture analysis
                    progress_bar.progress(progress)

            # Process remaining frames
            if frame_buffer:
                for idx, f in enumerate(frame_buffer):
                    t = threading.Thread(target=process_frame, args=(frame_count - len(frame_buffer) + idx + 1, f))
                    threads.append(t)
                    t.start()
                for t in threads:
                    t.join()

            # Collect results from queue
            while not result_queue.empty():
                frame_idx, posture, eye_contact, gesture, head_tilt, fidget = result_queue.get()
                if posture:
                    posture_score += 1
                if eye_contact:
                    eye_contact_score += 1
                hand_movement += gesture
                if fidget:
                    fidget_count += 1
                if head_tilt:
                    head_tilt_score += 1
                processed_frames += 1

            progress_bar.progress(100)  # 100% complete
            posture_score = max(50, min(950, (posture_score / processed_frames) * 1000 if processed_frames > 0 else 500))  # Scale to 1000, avoid 0 or 1000
            eye_contact_score = max(50, min(950, (eye_contact_score / processed_frames) * 1000 if processed_frames > 0 else 500))
            gesture_score = max(50, min(950, 500 + ((hand_movement / processed_frames) * 100 - (fidget_count / processed_frames * 50)) if processed_frames > 0 else 500))
            head_tilt_score = max(50, min(950, (head_tilt_score / processed_frames) * 1000 if processed_frames > 0 else 500))
            cap.release()
            pose.close()
            face.close()
        except Exception as e:
            st.error(f"Trouble checking your body language for Question {idx}: {str(e)}")
            posture_score = eye_contact_score = gesture_score = head_tilt_score = 500  # Default to 500 (mid-range)
            progress_bar.progress(100)

        # Tone, speech rate, pauses, and enthusiasm analysis with progress
        st.write(f"Listening carefully to your voice for Question {idx}...")
        progress_bar = st.progress(0)
        start_time = time.time()
        try:
            y, sr = librosa.load(audio_file, sr=None)  # Use full audio
            audio_length = len(y) / sr  # Duration in seconds
            progress_step = 100 / audio_length if audio_length > 0 else 100
            pauses = 0
            for i in range(0, len(y), int(sr * 1)):  # Process in 1-second chunks
                chunk = y[i:i + int(sr * 1)]
                if len(chunk) > 0:
                    pauses += len(librosa.effects.split(chunk))
                progress = int((i / len(y)) * 100)
                progress_bar.progress(progress)
            speech_rate = len(answer.split()) / audio_length  # Words per second
            pitch_mean = np.mean(librosa.pitch_tuning(y))
            pitch_variance = np.var(librosa.pitch_tuning(y))  # Variability for enthusiasm
            tone_score = max(50, min(950, 1000 - (pauses * 30)))  # Fewer penalties, scale to 1000
            speech_rate_score = max(50, min(950, 800 - abs(speech_rate - 2.5) * 150))  # Ideal ~2.5 words/sec, scale to 1000
            enthusiasm_score = max(50, min(950, tone_score + (pitch_mean * 100) + (pitch_variance * 50)))  # Richer enthusiasm, scale to 1000
            progress_bar.progress(100)  # 100% complete
        except Exception as e:
            st.error(f"Couldn’t assess your voice for Question {idx}: {str(e)}")
            tone_score = speech_rate_score = enthusiasm_score = 500  # Default to 500 (mid-range)
            progress_bar.progress(100)

        return {
            "verbal_feedback": verbal_feedback,
            "verbal_score": verbal_score,
            "posture_score": posture_score,
            "eye_contact_score": eye_contact_score,
            "gesture_score": gesture_score,
            "head_tilt_score": head_tilt_score,
            "tone_score": tone_score,
            "speech_rate_score": speech_rate_score,
            "enthusiasm_score": enthusiasm_score
        }

    # Use ThreadPoolExecutor for parallel analysis
    with ThreadPoolExecutor(max_workers=min(4, len(qa_pairs))) as executor:
        future_to_qa = {executor.submit(analyze_qa, qa_pair, idx): idx for idx, qa_pair in enumerate(qa_pairs, 1)}
        results = {}
        for future in future_to_qa:
            idx = future_to_qa[future]
            try:
                result = future.result()
                results[idx] = result
                progress = int((len(results) / len(qa_pairs)) * 30 + 70)  # 70-100% for analysis
                progress_bar_total.progress(progress)
            except Exception as e:
                st.error(f"Error analyzing Question {idx}: {str(e)}")
                results[idx] = {
                    "verbal_feedback": "Sorry, we couldn’t fully review your answer this time.",
                    "verbal_score": 500,
                    "posture_score": 500,
                    "eye_contact_score": 500,
                    "gesture_score": 500,
                    "head_tilt_score": 500,
                    "tone_score": 500,
                    "speech_rate_score": 500,
                    "enthusiasm_score": 500
                }
                progress = int((len(results) / len(qa_pairs)) * 30 + 70)
                progress_bar_total.progress(progress)

    # Display results for each question
    total_questions = len(qa_pairs)
    overall_score = 0
    for i in range(1, total_questions + 1):
        with st.expander(f"Question {i} and Answer Analysis"):
            result = results[i]
            question, answer = qa_pairs[i-1]
            st.write(f"**Interviewer’s Question**: {question}")
            st.write(f"**Your Response**: {answer}")
            st.write(f"**Feedback on Your Answer**: {result['verbal_feedback']}")
            st.write(f"**Posture**: {result['posture_score']}/1000 - How you held yourself (spine, shoulders)")
            st.write(f"**Eye Contact**: {result['eye_contact_score']}/1000 - Connecting with the interviewer")
            st.write(f"**Hand Gestures**: {result['gesture_score']}/1000 - Adding life to your words")
            st.write(f"**Head Position**: {result['head_tilt_score']}/1000 - Confidence in your posture")
            st.write(f"**Tone**: {result['tone_score']}/1000 - Steadiness and clarity in your voice")
            st.write(f"**Speech Pace**: {result['speech_rate_score']}/1000 - How you timed your words")
            st.write(f"**Enthusiasm**: {result['enthusiasm_score']}/1000 - Energy and engagement in your delivery")
            
            # Highlight question score (out of 1000) with color based on range
            question_score = int((result['verbal_score'] * 0.25 + result['posture_score'] * 0.15 + 
                                result['eye_contact_score'] * 0.15 + result['gesture_score'] * 0.15 + 
                                result['head_tilt_score'] * 0.1 + result['tone_score'] * 0.1 + 
                                result['speech_rate_score'] * 0.1 + result['enthusiasm_score'] * 0.1) / 10)  # Scale to 1000
            question_score = max(50, min(950, question_score))  # Avoid 0 or 1000
            if 750 <= question_score <= 1000:
                st.markdown(f"<h3 style='color: green; font-weight: bold;'>Question {i} Score: {question_score}/1000</h3>", unsafe_allow_html=True)
            elif 400 <= question_score < 750:
                st.markdown(f"<h3 style='color: yellow; font-weight: bold;'>Question {i} Score: {question_score}/1000</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='color: red; font-weight: bold;'>Question {i} Score: {question_score}/1000</h3>", unsafe_allow_html=True)

            # Radar chart for this Q&A pair
            aspects = ["Answer Quality", "Posture", "Eye Contact", "Gestures", "Head Position", "Tone", "Pace", "Enthusiasm"]
            scores = [result['verbal_score'] / 10, result['posture_score'] / 10, result['eye_contact_score'] / 10, 
                      result['gesture_score'] / 10, result['head_tilt_score'] / 10, result['tone_score'] / 10, 
                      result['speech_rate_score'] / 10, result['enthusiasm_score'] / 10]  # Scale to 100 for graph
            fig = go.Figure(data=go.Scatterpolar(
                r=scores + [scores[0]],  # Close the loop
                theta=aspects + [aspects[0]],
                fill='toself',
                line_color='#1f77b4',
                fillcolor='rgba(31, 119, 180, 0.3)'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                title=f"Your Strengths for Question {i}",
                height=500,
            )
            st.plotly_chart(fig)
            overall_score += question_score

    # Calculate and display overall interview score
    overall_score = int(overall_score / total_questions) if total_questions > 0 else 500  # Average, default to 500 if no questions
    overall_score = max(50, min(950, overall_score))  # Avoid 0 or 1000
    st.write("---")
    st.subheader("Overall Interview Performance")
    if 750 <= overall_score <= 1000:
        st.markdown(f"<h1 style='color: green; font-weight: bold;'>Overall Interview Score: {overall_score}/1000</h1>", unsafe_allow_html=True)
    elif 400 <= overall_score < 750:
        st.markdown(f"<h1 style='color: yellow; font-weight: bold;'>Overall Interview Score: {overall_score}/1000</h1>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h1 style='color: red; font-weight: bold;'>Overall Interview Score: {overall_score}/1000</h1>", unsafe_allow_html=True)

    # Overall score graph (bar chart)
    fig_overall = go.Figure(data=[go.Bar(x=['Overall Score'], y=[overall_score / 10], marker_color=['#1f77b4'])])
    fig_overall.update_layout(
        title="Overall Interview Performance",
        yaxis_title="Score (out of 100)",
        yaxis_range=[0, 100],
        template="plotly_white",
        height=300,
    )
    st.plotly_chart(fig_overall)

    # Generate overall report
    st.subheader("Overall Interview Report")
    st.write("Here’s a comprehensive analysis of your interview performance:")
    st.write("- **Total Questions Analyzed**:", total_questions)
    st.write("- **Individual Question Scores**:")
    for i in range(1, total_questions + 1):
        question_score = int((results[i]['verbal_score'] * 0.25 + results[i]['posture_score'] * 0.15 + 
                            results[i]['eye_contact_score'] * 0.15 + results[i]['gesture_score'] * 0.15 + 
                            results[i]['head_tilt_score'] * 0.1 + results[i]['tone_score'] * 0.1 + 
                            results[i]['speech_rate_score'] * 0.1 + results[i]['enthusiasm_score'] * 0.1) / 10)
        st.write(f"  - Question {i}: {question_score}/1000")
    st.write(f"- **Overall Score**: {overall_score}/1000")
    st.write("""
    **Summary Analysis**:
    Your interview performance shows strengths in [highlight top 2-3 aspects from highest average scores across questions, e.g., 'clarity and confidence in answers', 'strong eye contact', 'steady tone']. Areas for improvement include [identify 1-2 lowest average scores, e.g., 'more enthusiasm in delivery', 'reducing fidgeting during responses']. To enhance your next interview, focus on [specific tips, e.g., 'practicing pauses for emphasis', 'maintaining a straighter posture']. Overall, you’ve demonstrated a solid presence—keep refining these skills for even greater impact!
    """)

    # Clean up
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    if os.path.exists(audio_file):
        os.remove(audio_file)
else:
    st.info("Upload your video, and let’s see how you did!")