"""
============================================================
EXPERIMENT 10 - FINE-TUNING FOR DOMAIN ADAPTATION
============================================================

Course       : Applied Agentic AI
University   : Malla Reddy University
Domain       : Cybersecurity

Objective:
Train and evaluate a specialized model using synthetic
cybersecurity data for domain adaptation.

Workflow:

Synthetic Cybersecurity Dataset
              |
              v
       Pretrained Model
              |
              v
          Fine-Tuning
              |
              v
       Fine-Tuned Model
              |
              v
          Evaluation
              |
              v
      Accuracy Comparison

============================================================
"""

import os
import random
import numpy as np
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

OUTPUT_DIR = "./cybersecurity_finetuned_model"

NUM_LABELS = 3

LABEL_NAMES = {
    0: "BENIGN",
    1: "PHISHING",
    2: "MALWARE"
}


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ============================================================
# SYNTHETIC CYBERSECURITY DATASET
# ============================================================

training_data = [

    # BENIGN
    ("The employee successfully logged into the company portal.", 0),
    ("The user accessed the internal dashboard normally.", 0),
    ("The system completed a routine software update.", 0),
    ("The employee opened the approved company application.", 0),
    ("The server completed its scheduled backup.", 0),
    ("The user changed the password through the official portal.", 0),
    ("The employee accessed the company email normally.", 0),
    ("The workstation installed an approved security update.", 0),
    ("The administrator reviewed normal system logs.", 0),
    ("The employee accessed an authorized internal resource.", 0),

    # PHISHING
    ("The email asks the user to click a suspicious login link.", 1),
    ("The message requests the employee password through an unknown website.", 1),
    ("The email claims the account will be closed unless the user clicks a link.", 1),
    ("The attacker sends a fake banking login page to the employee.", 1),
    ("The message contains an urgent request for confidential credentials.", 1),
    ("The email impersonates the IT department and asks for a password.", 1),
    ("The suspicious message asks the employee to verify their account.", 1),
    ("The attacker uses a fake company login page to steal credentials.", 1),
    ("The email contains a malicious-looking authentication link.", 1),
    ("The message asks for sensitive information through an unknown domain.", 1),

    # MALWARE
    ("The downloaded file contains a malicious executable.", 2),
    ("The workstation is infected with ransomware.", 2),
    ("The system detects a trojan running in the background.", 2),
    ("The executable installs malicious software on the computer.", 2),
    ("The malware encrypts files and demands payment.", 2),
    ("The endpoint detects suspicious malicious code.", 2),
    ("The computer shows signs of a ransomware infection.", 2),
    ("A malicious program attempts to modify system files.", 2),
    ("The antivirus detects a trojan executable.", 2),
    ("The infected workstation connects to a suspicious command server.", 2)
]


evaluation_data = [

    ("The employee logged into the approved company application.", 0),
    ("The email asks the employee to enter credentials on a fake website.", 1),
    ("The computer was infected by ransomware that encrypted documents.", 2),
    ("The user accessed the normal internal company dashboard.", 0),
    ("The attacker sent an urgent email requesting the user's password.", 1),
    ("A malicious executable was detected on the workstation.", 2),
    ("The employee completed a routine system update.", 0),
    ("The message contains a suspicious link requesting account verification.", 1),
    ("The endpoint detected a trojan program.", 2)
]


# ============================================================
# CREATE DATASETS
# ============================================================

def create_datasets():

    train_dataset = Dataset.from_dict({
        "text": [
            item[0]
            for item in training_data
        ],
        "label": [
            item[1]
            for item in training_data
        ]
    })

    test_dataset = Dataset.from_dict({
        "text": [
            item[0]
            for item in evaluation_data
        ],
        "label": [
            item[1]
            for item in evaluation_data
        ]
    })

    return train_dataset, test_dataset


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_dataset(
    dataset,
    tokenizer
):

    return dataset.map(
        lambda examples: tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        ),
        batched=True
    )


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_prediction):

    predictions, labels = eval_prediction

    predictions = np.argmax(
        predictions,
        axis=1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    return {
        "accuracy": accuracy
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("     EXPERIMENT 10 - FINE-TUNING FOR DOMAIN ADAPTATION")
    print("=" * 70)

    print("\nCourse: Applied Agentic AI")
    print("University: Malla Reddy University")
    print("Domain: Cybersecurity")

    print("\nObjective:")
    print(
        "Train and evaluate a specialized model "
        "using synthetic cybersecurity data."
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("STEP 1 - SYNTHETIC DATASET")
    print("=" * 70)

    train_dataset, test_dataset = create_datasets()

    print(
        f"\nTraining samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Evaluation samples: "
        f"{len(test_dataset)}"
    )

    print("\nClasses:")

    for label, name in LABEL_NAMES.items():

        print(
            f"{label} -> {name}"
        )

    # --------------------------------------------------------
    # LOAD TOKENIZER
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("STEP 2 - LOADING PRETRAINED MODEL")
    print("=" * 70)

    print(
        f"\nModel: {MODEL_NAME}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=LABEL_NAMES,
        label2id={
            "BENIGN": 0,
            "PHISHING": 1,
            "MALWARE": 2
        }
    )

    print(
        "\nPretrained model loaded successfully."
    )

    # --------------------------------------------------------
    # TOKENIZE
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("STEP 3 - TOKENIZATION")
    print("=" * 70)

    tokenized_train = tokenize_dataset(
        train_dataset,
        tokenizer
    )

    tokenized_test = tokenize_dataset(
        test_dataset,
        tokenizer
    )

    tokenized_train = tokenized_train.remove_columns(
        ["text"]
    )

    tokenized_test = tokenized_test.remove_columns(
        ["text"]
    )

    tokenized_train.set_format("torch")
    tokenized_test.set_format("torch")

    print(
        "\nDataset tokenization completed."
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("STEP 4 - FINE-TUNING")
    print("=" * 70)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        num_train_epochs=3,

        per_device_train_batch_size=4,

        per_device_eval_batch_size=4,

        learning_rate=5e-5,

        weight_decay=0.01,

        eval_strategy="epoch",

        save_strategy="no",

        logging_strategy="epoch",

        report_to="none",

        fp16=False
    )

    trainer = Trainer(
        model=model,

        args=training_args,

        train_dataset=tokenized_train,

        eval_dataset=tokenized_test,

        compute_metrics=compute_metrics
    )

    print(
        "\nStarting fine-tuning..."
    )

    trainer.train()

    print(
        "\nFine-tuning completed successfully."
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("STEP 5 - MODEL EVALUATION")
    print("=" * 70)

    evaluation_results = trainer.evaluate()

    accuracy = evaluation_results.get(
        "eval_accuracy",
        0
    )

    print(
        f"\nEvaluation Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # TEST PREDICTIONS
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("STEP 6 - SAMPLE PREDICTIONS")
    print("=" * 70)

    sample_questions = [
        "The employee received a suspicious email asking for their password.",
        "The endpoint detected ransomware encrypting company files.",
        "The user accessed the approved internal company portal."
    ]

    for question in sample_questions:

        inputs = tokenizer(
            question,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )

        inputs = {
            key: value.to(model.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = model(
                **inputs
            )

        prediction = torch.argmax(
            outputs.logits,
            dim=1
        ).item()

        label = LABEL_NAMES[
            prediction
        ]

        print("\nInput:")
        print(question)

        print(
            f"Predicted Class: {label}"
        )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("STEP 7 - SAVE FINE-TUNED MODEL")
    print("=" * 70)

    trainer.save_model(
        OUTPUT_DIR
    )

    tokenizer.save_pretrained(
        OUTPUT_DIR
    )

    print(
        f"\nModel saved to: "
        f"{OUTPUT_DIR}"
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("EXPERIMENT 10 COMPLETED")
    print("=" * 70)

    print("\nWorkflow:")

    print(
        "1. Synthetic Dataset       - Created"
    )

    print(
        "2. Pretrained Model        - Loaded"
    )

    print(
        "3. Tokenization            - Completed"
    )

    print(
        "4. Fine-Tuning             - Completed"
    )

    print(
        "5. Model Evaluation        - Completed"
    )

    print(
        "6. Sample Predictions      - Completed"
    )

    print(
        "7. Fine-Tuned Model        - Saved"
    )

    print(
        "\nFinal Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        "\nFinal Status: SUCCESS"
    )

    print("\n" + "=" * 70)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()