"""
======================================================================
EXPERIMENT 11 - MODEL OPTIMIZATION
======================================================================

Course       : Applied Agentic AI
University   : Malla Reddy University
Domain       : Cybersecurity

Objective:
Apply dynamic INT8 quantization to a fine-tuned cybersecurity
classification model and compare the original and optimized models.

Workflow:

Fine-Tuned Model
       |
       v
Original Model
       |
       +--------------------+
       |                    |
       v                    v
Original Model        INT8 Quantized Model
       |                    |
       +----------+---------+
                  |
                  v
          Performance Comparison
                  |
       +----------+----------+
       |          |          |
       v          v          v
     Size      Accuracy    Speed

======================================================================
"""

import os
import time
import copy
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ======================================================================
# CONFIGURATION
# ======================================================================

MODEL_DIR = "../Experiment_10_Fine_Tuning_Domain_Adaptation/cybersecurity_finetuned_model"

OUTPUT_DIR = "./quantized_cybersecurity_model"

LABEL_NAMES = {
    0: "BENIGN",
    1: "PHISHING",
    2: "MALWARE"
}


# ======================================================================
# TEST DATA
# ======================================================================

test_data = [
    (
        "The employee received a suspicious email asking for their password.",
        1
    ),
    (
        "The endpoint detected ransomware encrypting company files.",
        2
    ),
    (
        "The user accessed the approved internal company portal.",
        0
    ),
    (
        "The attacker sent an urgent email requesting the user's password.",
        1
    ),
    (
        "A malicious executable was detected on the workstation.",
        2
    ),
    (
        "The employee completed a routine system update.",
        0
    ),
    (
        "The message contains a suspicious link requesting account verification.",
        1
    ),
    (
        "The endpoint detected a trojan program.",
        2
    ),
    (
        "The administrator reviewed normal system logs.",
        0
    )
]


# ======================================================================
# MODEL SIZE
# ======================================================================

def get_model_size(model):

    total_size = 0

    for parameter in model.parameters():
        total_size += parameter.numel() * parameter.element_size()

    for buffer in model.buffers():
        total_size += buffer.numel() * buffer.element_size()

    return total_size / (1024 * 1024)


# ======================================================================
# PREDICTION FUNCTION
# ======================================================================

def predict(model, tokenizer, text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = model(**inputs)

    prediction = torch.argmax(
        outputs.logits,
        dim=1
    ).item()

    return prediction


# ======================================================================
# EVALUATION
# ======================================================================

def evaluate_model(model, tokenizer):

    correct = 0

    start_time = time.perf_counter()

    predictions = []

    for text, true_label in test_data:

        prediction = predict(
            model,
            tokenizer,
            text
        )

        predictions.append(prediction)

        if prediction == true_label:
            correct += 1

    end_time = time.perf_counter()

    accuracy = correct / len(test_data)

    inference_time = end_time - start_time

    return accuracy, inference_time, predictions


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("\n")
    print("=" * 70)
    print("        EXPERIMENT 11 - MODEL OPTIMIZATION")
    print("=" * 70)

    print("\nCourse: Applied Agentic AI")
    print("University: Malla Reddy University")
    print("Domain: Cybersecurity")

    print("\nObjective:")
    print(
        "Apply dynamic INT8 quantization to optimize "
        "a fine-tuned cybersecurity classification model."
    )

    # ------------------------------------------------------------------
    # LOAD MODEL
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("STEP 1 - LOAD FINE-TUNED MODEL")
    print("=" * 70)

    print(
        f"\nModel path:\n{MODEL_DIR}"
    )

    if not os.path.exists(MODEL_DIR):

        print(
            "\nERROR: Fine-tuned model not found."
        )

        print(
            "\nMake sure Experiment 10 has been completed first."
        )

        return

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    )

    model.eval()

    print(
        "\nFine-tuned model loaded successfully."
    )

    # ------------------------------------------------------------------
    # ORIGINAL MODEL SIZE
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("STEP 2 - ORIGINAL MODEL")
    print("=" * 70)

    original_size = get_model_size(model)

    print(
        f"\nOriginal model size: "
        f"{original_size:.2f} MB"
    )

    # ------------------------------------------------------------------
    # ORIGINAL MODEL EVALUATION
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("STEP 3 - ORIGINAL MODEL EVALUATION")
    print("=" * 70)

    original_accuracy, original_time, original_predictions = evaluate_model(
        model,
        tokenizer
    )

    print(
        f"\nOriginal Accuracy: "
        f"{original_accuracy * 100:.2f}%"
    )

    print(
        f"Original Inference Time: "
        f"{original_time:.4f} seconds"
    )

    # ------------------------------------------------------------------
    # QUANTIZATION
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("STEP 4 - INT8 QUANTIZATION")
    print("=" * 70)

    print(
        "\nApplying dynamic INT8 quantization..."
    )

    quantized_model = copy.deepcopy(model)

    quantized_model = torch.quantization.quantize_dynamic(
        quantized_model,
        {
            torch.nn.Linear
        },
        dtype=torch.qint8
    )

    quantized_model.eval()

    print(
        "\nINT8 quantization completed successfully."
    )

    # ------------------------------------------------------------------
    # QUANTIZED MODEL SIZE
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("STEP 5 - QUANTIZED MODEL")
    print("=" * 70)

    quantized_size = get_model_size(
        quantized_model
    )

    print(
        f"\nQuantized model size: "
        f"{quantized_size:.2f} MB"
    )

    # ------------------------------------------------------------------
    # QUANTIZED MODEL EVALUATION
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("STEP 6 - QUANTIZED MODEL EVALUATION")
    print("=" * 70)

    quantized_accuracy, quantized_time, quantized_predictions = evaluate_model(
        quantized_model,
        tokenizer
    )

    print(
        f"\nQuantized Accuracy: "
        f"{quantized_accuracy * 100:.2f}%"
    )

    print(
        f"Quantized Inference Time: "
        f"{quantized_time:.4f} seconds"
    )

    # ------------------------------------------------------------------
    # PERFORMANCE COMPARISON
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("STEP 7 - PERFORMANCE COMPARISON")
    print("=" * 70)

    if original_size > 0:

        size_reduction = (
            (original_size - quantized_size)
            / original_size
        ) * 100

    else:

        size_reduction = 0

    accuracy_difference = (
        quantized_accuracy - original_accuracy
    ) * 100

    if quantized_time > 0:

        speed_ratio = (
            original_time / quantized_time
        )

    else:

        speed_ratio = 0

    print(
        f"\nOriginal Model Size   : "
        f"{original_size:.2f} MB"
    )

    print(
        f"Quantized Model Size  : "
        f"{quantized_size:.2f} MB"
    )

    print(
        f"Size Reduction        : "
        f"{size_reduction:.2f}%"
    )

    print(
        f"\nOriginal Accuracy     : "
        f"{original_accuracy * 100:.2f}%"
    )

    print(
        f"Quantized Accuracy    : "
        f"{quantized_accuracy * 100:.2f}%"
    )

    print(
        f"Accuracy Difference   : "
        f"{accuracy_difference:+.2f}%"
    )

    print(
        f"\nOriginal Inference    : "
        f"{original_time:.4f} seconds"
    )

    print(
        f"Quantized Inference   : "
        f"{quantized_time:.4f} seconds"
    )

    print(
        f"Speed Ratio           : "
        f"{speed_ratio:.2f}x"
    )

    # ------------------------------------------------------------------
    # SAMPLE PREDICTIONS
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("STEP 8 - SAMPLE PREDICTIONS")
    print("=" * 70)

    sample_questions = [
        "The employee received a suspicious email asking for their password.",
        "The endpoint detected ransomware encrypting company files.",
        "The user accessed the approved internal company portal."
    ]

    for question in sample_questions:

        original_prediction = predict(
            model,
            tokenizer,
            question
        )

        quantized_prediction = predict(
            quantized_model,
            tokenizer,
            question
        )

        print("\nInput:")
        print(question)

        print(
            f"Original Model  : "
            f"{LABEL_NAMES[original_prediction]}"
        )

        print(
            f"Quantized Model : "
            f"{LABEL_NAMES[quantized_prediction]}"
        )

    # ------------------------------------------------------------------
    # SAVE QUANTIZED MODEL INFORMATION
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("STEP 9 - SAVE OPTIMIZED MODEL")
    print("=" * 70)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # Save tokenizer
    tokenizer.save_pretrained(
        OUTPUT_DIR
    )

    # Save quantized model state
    torch.save(
        quantized_model.state_dict(),
        os.path.join(
            OUTPUT_DIR,
            "quantized_model_state_dict.pth"
        )
    )

    # Save optimization information
    with open(
        os.path.join(
            OUTPUT_DIR,
            "optimization_results.txt"
        ),
        "w"
    ) as file:

        file.write(
            "EXPERIMENT 11 - MODEL OPTIMIZATION\n"
        )

        file.write(
            "===================================\n\n"
        )

        file.write(
            f"Original Model Size: {original_size:.2f} MB\n"
        )

        file.write(
            f"Quantized Model Size: {quantized_size:.2f} MB\n"
        )

        file.write(
            f"Size Reduction: {size_reduction:.2f}%\n"
        )

        file.write(
            f"Original Accuracy: {original_accuracy * 100:.2f}%\n"
        )

        file.write(
            f"Quantized Accuracy: {quantized_accuracy * 100:.2f}%\n"
        )

        file.write(
            f"Original Inference Time: {original_time:.4f} seconds\n"
        )

        file.write(
            f"Quantized Inference Time: {quantized_time:.4f} seconds\n"
        )

    print(
        f"\nOptimized model saved to:"
        f"\n{OUTPUT_DIR}"
    )

    # ------------------------------------------------------------------
    # FINAL RESULT
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("EXPERIMENT 11 COMPLETED")
    print("=" * 70)

    print("\nWorkflow:")

    print(
        "1. Fine-Tuned Model       - Loaded"
    )

    print(
        "2. Original Model        - Evaluated"
    )

    print(
        "3. INT8 Quantization     - Completed"
    )

    print(
        "4. Quantized Model       - Evaluated"
    )

    print(
        "5. Performance Comparison- Completed"
    )

    print(
        "6. Sample Predictions    - Completed"
    )

    print(
        "7. Optimized Model       - Saved"
    )

    print(
        "\nFinal Status: SUCCESS"
    )

    print("\n" + "=" * 70)


# ======================================================================
# PROGRAM ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()