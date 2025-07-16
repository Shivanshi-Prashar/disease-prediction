import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import mysql.connector
from ultralytics import YOLO
import cv2

# ------------------- Model Paths -------------------
ACNE_MODEL_PATH = "disease_float_final.tflite"
TYPE_MODEL_PATH = 'C:/webdevelopment/Cure-Catalyst/Project/ml_server/artifacts/training/2_class_final_6feb.tflite'

# ------------------- Load Models -------------------
@st.cache_resource
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

@st.cache_resource
def load_yolo_model():
    return YOLO("C:/Acne/runs/runs/detect/train4/weights/best.pt")  # update path if needed

acne_interpreter = load_tflite_model(ACNE_MODEL_PATH)
type_interpreter = load_tflite_model(TYPE_MODEL_PATH)
yolo_model = load_yolo_model()

# ------------------- Image Preprocessing -------------------
def preprocess_image(image, target_size=(380, 380)):
    image = image.resize(target_size)
    image = np.array(image, dtype=np.float32)
    image = np.expand_dims(image, axis=0)
    return image

# ------------------- TFLite Inference -------------------
def run_tflite_model(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

# ------------------- Acne and Type Prediction -------------------
def predict_acne(image):
    image_array = preprocess_image(image)
    probability = run_tflite_model(acne_interpreter, image_array)[0][0]
    result = "Acne Detected" if probability <= 0.5 else "No Acne"
    probability = abs((0.5 - probability) * 200)
    return probability, result

def predict_type(image):
    image_array = preprocess_image(image)
    type_probability = run_tflite_model(type_interpreter, image_array)[0][0]
    result = "Inflammatory Acne" if type_probability <= 0.5 else "Non-Inflammatory Acne"
    type_probability = abs((0.5 - type_probability) * 200)
    return type_probability, result

# ------------------- YOLO Object Detection + Grade Mapping -------------------
import numpy as np
import cv2

def detect_and_count_lesions(image):
    results = yolo_model(image, conf=0.15)
    boxes = results[0].boxes
    class_ids = boxes.cls.cpu().numpy().astype(int)
    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    class_names = yolo_model.names

    count_dict = {}

    # Convert input image to NumPy BGR format (for OpenCV)
    if isinstance(image, np.ndarray):
        image_bgr = image.copy()
    else:
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for cid, box in zip(class_ids, xyxy):
        label = class_names[cid]
        count_dict[label] = count_dict.get(label, 0) + 1

        x1, y1, x2, y2 = box
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_bgr, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    result_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return count_dict, result_rgb


def map_counts_to_severity_grade(counts):
    total_lesions = sum(counts.values())
    nodules = counts.get('nodules', 0)
    pustules = counts.get('pustules',0)
    papules = counts.get('papules',0)
    if nodules > 15 or total_lesions >= 50:
        return 4
    elif nodules >= 5 or total_lesions >= 30:
        return 3
    elif nodules>=1 or pustules >=1 or papules>=2 or total_lesions >= 10:
        return 2
    else:
        return 1

# ------------------- Questionnaire Utilities -------------------
def load_questions():
    with open("questions.json", "r") as file:
        return json.load(file)["questions"]

def get_unique_questions(grades):
    all_questions = load_questions()
    selected_questions = {}
    for q in all_questions:
        for grade in grades:
            if grade in q["grades"]:
                selected_questions[q["question"]] = q
    return list(selected_questions.values())

def combined_acne_assessment_form(grades):
    st.subheader(f"Acne Assessment Form")
    questions = get_unique_questions(grades)
    answers = {}
    form_key = f'combined_acne_form_{",".join(map(str, grades))}'
    with st.form(key=form_key):
        for question in questions:
            answer = st.radio(question["question"], question["options"], key=question["question"])
            answers[question["question"]] = answer
        submit_button = st.form_submit_button(label="Submit")
    if submit_button:
        return calculate_grade_scores(grades, questions, answers)
    return []

def calculate_grade_scores(grades, questions, answers):
    scores = {1: 0, 2: 0, 3: 0, 4: 0}
    for grade in scores:
        total_weight = sum(q["weight"] for q in questions if grade in q["grades"])
        weighted_score = 0
        for q in questions:
            if grade in q["grades"]:
                correct_answer = q["answers"].get(str(grade))
                if correct_answer and answers.get(q["question"]) == correct_answer:
                    weighted_score += q["weight"]
        scores[grade] = (weighted_score / total_weight) * 100 if total_weight > 0 else 0
    return scores

def get_top_two_grades(severity_percentages,type_result):
    highest_grade = max(severity_percentages, key=severity_percentages.get)

    if highest_grade == 1:
        return [(1, severity_percentages.get(1, 0)), (2, severity_percentages.get(2, 0))]
    
    elif highest_grade == 2:
        if type_result=="Non-Inflammatory Acne":
            return [(1, severity_percentages.get(1, 0)), (2, severity_percentages.get(2, 0))]
        else:
            return [(2, severity_percentages.get(2, 0)), (3, severity_percentages.get(3, 0))]

    elif highest_grade == 3:
        if severity_percentages.get(2, 0) >= severity_percentages.get(4, 0):
            return [(2, severity_percentages.get(2, 0)), (3, severity_percentages.get(3, 0))]
        else:
            return [(3, severity_percentages.get(3, 0)), (4, severity_percentages.get(4, 0))]

    elif highest_grade == 4:
        return [(3, severity_percentages.get(3, 0)), (4, severity_percentages.get(4, 0))]

# ------------------- Database Treatment Query -------------------
def get_treatment(subtype_name, severity, disease_name):
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='kunal122604',
            database='acne'
        )
        cursor = conn.cursor(dictionary=True)
        query = """
        SELECT 
            md.Medicine,
            md.Category,
            sd.Precautions,
            sd.Remedies
        FROM 
            treatment_severity ts
        JOIN 
            subtype_disease sd ON ts.SubtypeID = sd.SubtypeId
        JOIN 
            disease d ON d.DiseaseID = sd.DiseaseID
        JOIN 
            medication_description md ON ts.MedicineID = md.MedicineID
        WHERE 
            sd.Subtype_Name = %s 
            AND ts.Severity = %s 
            AND d.Disease_Name = %s
        LIMIT 1;
        """
        cursor.execute(query, (subtype_name, severity, disease_name))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return result if result else "No matching record found"
    except mysql.connector.Error as err:
        return f"Error: {err}"

# ------------------- Streamlit App -------------------
st.title("Acne Detection, Type Classification, and Severity Grading")

uploaded_files = st.file_uploader("Upload Images (Min: 1, Max: 6)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) > 6:
        st.error("You can upload a maximum of 6 images.")
    else:
        st.success(f"{len(uploaded_files)} image(s) uploaded.")
        cols = st.columns(6)
        images_to_predict = []
        results = []
        acne_prob, non_acne_prob = 0, 0
        infla_prob, non_infla_prob = 0, 0
        severity_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        finalGrade_score = {1: 0, 2: 0, 3: 0, 4: 0}

        for i, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file).convert("RGB")
            probability, acne_result = predict_acne(image)
            st.write(f"Processing image {i + 1}: {acne_result} ({probability:.2f}%)")

            if acne_result == "Acne Detected":
                images_to_predict.append(image)
                acne_prob += probability
                cols[i % 6].image(image, caption=f"{acne_result} ({probability:.2f}%)", use_container_width=True)
            else:
                non_acne_prob += probability

            

        acne_prob /= max(1, len(uploaded_files))
        non_acne_prob /= max(1, len(uploaded_files))
        final_result = "Acne Detected" if acne_prob > non_acne_prob else "No Acne"
        final_prob = max(acne_prob, non_acne_prob)

        st.write(f"**Average Probability:** {final_prob:.2f}")
        st.write(f"**Final Result:** {final_result}")

        if final_result == "Acne Detected":
            st.write("### Running further models...")
            severity_cols = st.columns(6)

            for i, image in enumerate(images_to_predict):
                type_probability, type_result = predict_type(image)
                detection_counts, detection_plot = detect_and_count_lesions(image)
                assigned_grade = map_counts_to_severity_grade(detection_counts)
                if assigned_grade > 2:
                    type_result =  "Inflammatory Acne"

                results.append((image, type_probability, type_result, assigned_grade, 100.0))

                infla_prob += type_probability if type_result == "Inflammatory Acne" else 0
                non_infla_prob += type_probability if type_result == "Non-Inflammatory Acne" else 0

                severity_counts[assigned_grade] += 100

                st.write(f"Image {i+1} Detection:")
                st.write(f"Lesions Detected: {detection_counts}")
                st.write(f"Assigned Grade: {assigned_grade}")
                severity_cols[i % 6].image(detection_plot, caption=f"Grade {assigned_grade}, Type: {type_result}", use_container_width=True)

            severity_percentages = {grade: (count / len(images_to_predict)) for grade, count in severity_counts.items()}

            #st.write("### Severity Distribution:")
            #for grade, percentage in severity_percentages.items():
            #    st.write(f"Grade {grade}: {percentage:.2f}%")

            top_two_grades = get_top_two_grades(severity_percentages,type_result)
            top_grades = [grade for grade, _ in top_two_grades]

            score = combined_acne_assessment_form(top_grades)

            if score:
                for grade in finalGrade_score:
                    finalGrade_score[grade] = (severity_percentages[grade] + score[grade]) / 2

                final_grade_result = max(finalGrade_score, key=finalGrade_score.get)
                final_grade_probability = finalGrade_score[final_grade_result]

                final_type_result = "INFLAMMATORY ACNE" if infla_prob > non_infla_prob else "NON-INFLAMMATORY ACNE"
                final_type_score = max(infla_prob, non_infla_prob) / len(images_to_predict)

                st.write(f"**Final Acne Grade:** {final_grade_result} **with percentage:** {final_grade_probability}")
                st.write(f"**Acne Type:** {final_type_result}")

                predicted_disease = "Acne Vulgaris"
                predicted_type = "Inflammatory acne" if final_type_result == "INFLAMMATORY ACNE" else "Non-inflammatory acne"
                predicted_severity = "Mild" if final_grade_result == 1 else "Moderate " if final_grade_result == 2 else "Severe"

                record = get_treatment(predicted_type, predicted_severity, predicted_disease)
                if isinstance(record, dict):
                    st.markdown(f"**Medicine:** {record['Medicine']}")
                    st.markdown(f"**Category:** {record['Category']}")
                    st.markdown(f"**Precautions:** {record['Precautions']}")
                    st.markdown(f"**Remedies:** {record['Remedies']}")
                else:
                    st.warning(record)
