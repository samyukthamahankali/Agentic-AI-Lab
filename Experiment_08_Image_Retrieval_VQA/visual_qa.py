"""
============================================================
EXPERIMENT 8 - IMAGE RETRIEVAL / VISUAL QA SYSTEM
============================================================

Course       : Applied Agentic AI
University   : Malla Reddy University
Model        : llama3.2:3b via Ollama

Objective:
Develop an image retrieval and visual question answering
pipeline using image processing and an LLM.

Pipeline:

Input Image
     |
     v
Image Processing
     |
     v
Image Information Extraction
     |
     v
User Question
     |
     v
Llama 3.2
     |
     v
Visual QA Answer

Note:
The text-only Llama 3.2 model does not directly process pixels.
The image is analyzed using Python/Pillow and the extracted
information is supplied to the LLM.
============================================================
"""

import os
import requests
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"
OLLAMA_TIMEOUT = 120


# ============================================================
# OLLAMA FUNCTION
# ============================================================

def ask_llama(prompt):
    """
    Send a prompt to the local Llama model through Ollama.
    """

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 200
        }
    }

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=OLLAMA_TIMEOUT
        )

        response.raise_for_status()

        result = response.json()

        answer = result.get("response", "").strip()

        if not answer:
            return "ERROR: Empty response from Ollama."

        return answer

    except requests.exceptions.ConnectionError:

        return (
            "ERROR: Could not connect to Ollama. "
            "Make sure Ollama is running."
        )

    except requests.exceptions.Timeout:

        return "ERROR: Ollama request timed out."

    except Exception as error:

        return f"ERROR: {error}"


# ============================================================
# IMAGE INFORMATION EXTRACTION
# ============================================================

def analyze_image(image_path):
    """
    Extract basic information from an image using Pillow.
    """

    print("\n")
    print("=" * 70)
    print("IMAGE PROCESSING")
    print("=" * 70)

    if not os.path.exists(image_path):

        return {
            "error": "Image file not found."
        }

    try:

        image = Image.open(image_path)

        width, height = image.size

        image_format = image.format

        color_mode = image.mode

        file_size = os.path.getsize(image_path)

        file_size_kb = round(
            file_size / 1024,
            2
        )

        information = {
            "filename": os.path.basename(image_path),
            "format": image_format,
            "width": width,
            "height": height,
            "color_mode": color_mode,
            "file_size_kb": file_size_kb
        }

        print("\nImage Information:")

        print(
            f"Filename     : {information['filename']}"
        )

        print(
            f"Format       : {information['format']}"
        )

        print(
            f"Dimensions   : "
            f"{information['width']} x "
            f"{information['height']}"
        )

        print(
            f"Color Mode   : {information['color_mode']}"
        )

        print(
            f"File Size    : "
            f"{information['file_size_kb']} KB"
        )

        return information

    except Exception as error:

        return {
            "error": str(error)
        }


# ============================================================
# IMAGE RETRIEVAL
# ============================================================

def retrieve_image_information(image_information, question):
    """
    Retrieve information relevant to the user's question.

    This lightweight version uses image metadata because the
    installed Llama 3.2 model is text-only.
    """

    print("\n")
    print("=" * 70)
    print("IMAGE RETRIEVAL")
    print("=" * 70)

    relevant_information = {
        "filename": image_information.get("filename"),
        "format": image_information.get("format"),
        "dimensions": (
            f"{image_information.get('width')} x "
            f"{image_information.get('height')}"
        ),
        "color_mode": image_information.get("color_mode"),
        "file_size_kb": image_information.get("file_size_kb")
    }

    print("\nQuestion:")
    print(question)

    print("\nRetrieved Image Information:")
    print(relevant_information)

    return relevant_information


# ============================================================
# VISUAL QA AGENT
# ============================================================

def visual_qa_agent(question, image_information):
    """
    Generate an answer using the retrieved image information.
    """

    print("\n")
    print("=" * 70)
    print("VISUAL QA AGENT")
    print("=" * 70)

    prompt = f"""
You are a Visual Question Answering agent.

The original image was processed by an image-processing
component. The available image information is:

Filename: {image_information.get('filename')}
Format: {image_information.get('format')}
Dimensions: {image_information.get('dimensions')}
Color Mode: {image_information.get('color_mode')}
File Size: {image_information.get('file_size_kb')} KB

User Question:
{question}

Answer the question using ONLY the available image
information.

If the available information is not sufficient to answer
the question, clearly say:

"Insufficient visual information available."

Do not invent objects, people, colors, locations, or actions.

Give a short and clear answer.
"""

    answer = ask_llama(prompt)

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(answer)

    return answer


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("      EXPERIMENT 8 - IMAGE RETRIEVAL / VISUAL QA")
    print("=" * 70)

    print("\nCourse: Applied Agentic AI")
    print("University: Malla Reddy University")
    print("Model:", MODEL_NAME)

    print("\nObjective:")
    print(
        "Develop an image retrieval and visual "
        "question answering pipeline."
    )

    # --------------------------------------------------------
    # IMAGE INPUT
    # --------------------------------------------------------

    image_path = input(
        "\nEnter image path: "
    ).strip()

    if not image_path:

        print("\nERROR: Image path cannot be empty.")
        return

    # --------------------------------------------------------
    # PROCESS IMAGE
    # --------------------------------------------------------

    image_information = analyze_image(
        image_path
    )

    if "error" in image_information:

        print(
            "\nERROR:",
            image_information["error"]
        )

        return

    # --------------------------------------------------------
    # QUESTION INPUT
    # --------------------------------------------------------

    question = input(
        "\nEnter your question about the image: "
    ).strip()

    if not question:

        print("\nERROR: Question cannot be empty.")
        return

    # --------------------------------------------------------
    # IMAGE RETRIEVAL
    # --------------------------------------------------------

    retrieved_information = retrieve_image_information(
        image_information,
        question
    )

    # --------------------------------------------------------
    # VISUAL QA
    # --------------------------------------------------------

    answer = visual_qa_agent(
        question,
        retrieved_information
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("EXPERIMENT 8 COMPLETED")
    print("=" * 70)

    print("\nPipeline:")
    print("1. Image Input              - Completed")
    print("2. Image Processing         - Completed")
    print("3. Image Information       - Retrieved")
    print("4. Visual QA Agent          - Completed")

    print("\nFinal Answer:")
    print(answer)

    print("\nFinal Status: SUCCESS")

    print("\n" + "=" * 70)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
