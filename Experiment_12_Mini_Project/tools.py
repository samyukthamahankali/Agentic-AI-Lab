"""
======================================================================
CYBERSHIELD AI - SECURITY TOOLS
======================================================================

Project      : CyberShield AI
Course       : Applied Agentic AI
University   : Malla Reddy University
Domain       : Cybersecurity

Tools:
1. Phishing Indicator Analyzer
2. Password Strength Analyzer
3. IP Address Analyzer
4. Security Severity Classifier

======================================================================
"""

import re
import ipaddress


# ======================================================================
# 1. PHISHING INDICATOR ANALYZER
# ======================================================================

def analyze_phishing(text):

    # Convert input to lowercase
    text = text.lower().strip()

    # Normalize spaces
    text = re.sub(r"\s+", " ", text)

    indicators = []

    # --------------------------------------------------------------
    # Suspicious phrases
    # --------------------------------------------------------------

    suspicious_phrases = [
        "urgent",
        "immediately",
        "verify your account",
        "verify my account",
        "verify account",
        "confirm your password",
        "confirm my password",
        "provide your password",
        "provide my password",
        "password",
        "click here",
        "login",
        "log in",
        "account verification",
        "account verify",
        "reset your password",
        "reset my password",
        "update your account",
        "update my account",
        "suspicious link",
        "security alert",
        "act now",
        "immediate action",
        "limited time",
        "credentials",
        "credential",
        "banking information",
        "financial information"
    ]

    # --------------------------------------------------------------
    # Detect phrases
    # --------------------------------------------------------------

    for phrase in suspicious_phrases:

        if phrase in text:

            indicators.append(
                f"Suspicious phrase detected: '{phrase}'"
            )

    # --------------------------------------------------------------
    # Suspicious URL indicators
    # --------------------------------------------------------------

    if "http://" in text:

        indicators.append(
            "Unencrypted HTTP link detected"
        )

    if "https://" in text:

        indicators.append(
            "External HTTPS link detected"
        )

    # --------------------------------------------------------------
    # Shortened URL services
    # --------------------------------------------------------------

    shortened_domains = [
        "bit.ly",
        "tinyurl.com",
        "t.co",
        "goo.gl",
        "ow.ly"
    ]

    for domain in shortened_domains:

        if domain in text:

            indicators.append(
                f"Shortened URL detected: '{domain}'"
            )

    # --------------------------------------------------------------
    # Suspicious email patterns
    # --------------------------------------------------------------

    if "dear customer" in text:

        indicators.append(
            "Generic greeting detected"
        )

    if "dear user" in text:

        indicators.append(
            "Generic greeting detected"
        )

    # --------------------------------------------------------------
    # Determine risk
    # --------------------------------------------------------------

    count = len(indicators)

    if count >= 5:

        risk = "HIGH"

    elif count >= 2:

        risk = "MEDIUM"

    elif count == 1:

        risk = "LOW"

    else:

        risk = "BENIGN"

    return {
        "tool": "Phishing Indicator Analyzer",
        "risk": risk,
        "indicators": indicators,
        "indicator_count": count
    }


# ======================================================================
# 2. PASSWORD STRENGTH ANALYZER
# ======================================================================

def analyze_password(password):

    # Remove accidental whitespace around password
    password = password.strip()

    score = 0
    feedback = []

    # --------------------------------------------------------------
    # Length
    # --------------------------------------------------------------

    if len(password) >= 8:

        score += 1

    else:

        feedback.append(
            "Password should contain at least 8 characters."
        )

    # --------------------------------------------------------------
    # Uppercase
    # --------------------------------------------------------------

    if re.search(r"[A-Z]", password):

        score += 1

    else:

        feedback.append(
            "Add at least one uppercase letter."
        )

    # --------------------------------------------------------------
    # Lowercase
    # --------------------------------------------------------------

    if re.search(r"[a-z]", password):

        score += 1

    else:

        feedback.append(
            "Add at least one lowercase letter."
        )

    # --------------------------------------------------------------
    # Number
    # --------------------------------------------------------------

    if re.search(r"\d", password):

        score += 1

    else:

        feedback.append(
            "Add at least one number."
        )

    # --------------------------------------------------------------
    # Special character
    # --------------------------------------------------------------

    if re.search(r"[^A-Za-z0-9]", password):

        score += 1

    else:

        feedback.append(
            "Add at least one special character."
        )

    # --------------------------------------------------------------
    # Strength
    # --------------------------------------------------------------

    if score == 5:

        strength = "STRONG"

    elif score >= 3:

        strength = "MEDIUM"

    else:

        strength = "WEAK"

    return {
        "tool": "Password Strength Analyzer",
        "strength": strength,
        "score": f"{score}/5",
        "feedback": feedback
    }


# ======================================================================
# 3. IP ADDRESS ANALYZER
# ======================================================================

def analyze_ip_address(ip):

    # --------------------------------------------------------------
    # Clean the supplied IP address
    # --------------------------------------------------------------

    ip = str(ip).strip()

    # Remove punctuation accidentally attached to the IP
    ip = ip.strip(
        " \t\r\n.,:;!?()[]{}<>\"'"
    )

    # --------------------------------------------------------------
    # Validate IP
    # --------------------------------------------------------------

    try:

        ip_object = ipaddress.ip_address(ip)

    except ValueError:

        return {
            "tool": "IP Address Analyzer",
            "ip": ip,
            "classification": "INVALID",
            "explanation": "The supplied value is not a valid IP address."
        }

    # --------------------------------------------------------------
    # Loopback
    # --------------------------------------------------------------

    if ip_object.is_loopback:

        return {
            "tool": "IP Address Analyzer",
            "ip": ip,
            "classification": "LOOPBACK",
            "explanation":
                "This IP address refers to the local host itself."
        }

    # --------------------------------------------------------------
    # Private
    # --------------------------------------------------------------

    if ip_object.is_private:

        return {
            "tool": "IP Address Analyzer",
            "ip": ip,
            "classification": "PRIVATE",
            "explanation":
                "This IP address belongs to a private network range."
        }

    # --------------------------------------------------------------
    # Link-local
    # --------------------------------------------------------------

    if ip_object.is_link_local:

        return {
            "tool": "IP Address Analyzer",
            "ip": ip,
            "classification": "LINK-LOCAL",
            "explanation":
                "This IP address belongs to a link-local address range."
        }

    # --------------------------------------------------------------
    # Multicast
    # --------------------------------------------------------------

    if ip_object.is_multicast:

        return {
            "tool": "IP Address Analyzer",
            "ip": ip,
            "classification": "MULTICAST",
            "explanation":
                "This IP address belongs to a multicast address range."
        }

    # --------------------------------------------------------------
    # Public
    # --------------------------------------------------------------

    return {
        "tool": "IP Address Analyzer",
        "ip": ip,
        "classification": "PUBLIC",
        "explanation":
            "This IP address is not in a private or loopback range."
    }


# ======================================================================
# 4. SECURITY SEVERITY CLASSIFIER
# ======================================================================

def classify_severity(text):

    text = str(text).lower().strip()

    high_risk_indicators = [
        "ransomware",
        "encrypted files",
        "data exfiltration",
        "critical vulnerability",
        "system compromise",
        "account takeover",
        "malware infection",
        "trojan",
        "remote access",
        "unauthorized access",
        "sensitive data breach",
        "data breach"
    ]

    medium_risk_indicators = [
        "failed login",
        "multiple failed login attempts",
        "suspicious email",
        "phishing",
        "suspicious link",
        "unusual activity",
        "suspicious activity",
        "malicious attachment",
        "security alert"
    ]

    high_found = []
    medium_found = []

    # --------------------------------------------------------------
    # HIGH indicators
    # --------------------------------------------------------------

    for indicator in high_risk_indicators:

        if indicator in text:

            high_found.append(
                indicator
            )

    # --------------------------------------------------------------
    # MEDIUM indicators
    # --------------------------------------------------------------

    for indicator in medium_risk_indicators:

        if indicator in text:

            medium_found.append(
                indicator
            )

    # --------------------------------------------------------------
    # Severity decision
    # --------------------------------------------------------------

    if high_found:

        severity = "HIGH"

    elif medium_found:

        severity = "MEDIUM"

    else:

        severity = "LOW"

    return {
        "tool": "Security Severity Classifier",
        "severity": severity,
        "high_risk_indicators": high_found,
        "medium_risk_indicators": medium_found
    }


# ======================================================================
# TEST PROGRAM
# ======================================================================

def main():

    print("\n")
    print("=" * 70)
    print("             CYBERSHIELD AI - SECURITY TOOLS")
    print("=" * 70)

    # --------------------------------------------------------------
    # PHISHING TEST
    # --------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("1. PHISHING ANALYZER")
    print("=" * 70)

    phishing_text = (
        "URGENT! Verify your account immediately. "
        "Click here to login and confirm your password."
    )

    result = analyze_phishing(
        phishing_text
    )

    print("\nInput:")
    print(phishing_text)

    print("\nResult:")
    print(
        f"Risk: {result['risk']}"
    )

    print(
        f"Indicators found: "
        f"{result['indicator_count']}"
    )

    for indicator in result["indicators"]:

        print(
            f"- {indicator}"
        )

    # --------------------------------------------------------------
    # PASSWORD TEST
    # --------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("2. PASSWORD ANALYZER")
    print("=" * 70)

    password = "CyberShield@2026"

    result = analyze_password(
        password
    )

    print(
        f"\nPassword: {password}"
    )

    print(
        f"Strength: {result['strength']}"
    )

    print(
        f"Score: {result['score']}"
    )

    if result["feedback"]:

        for item in result["feedback"]:

            print(
                f"- {item}"
            )

    # --------------------------------------------------------------
    # IP TESTS
    # --------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("3. IP ADDRESS ANALYZER")
    print("=" * 70)

    test_ips = [
        "192.168.1.10",
        "8.8.8.8",
        "127.0.0.1"
    ]

    for ip in test_ips:

        result = analyze_ip_address(
            ip
        )

        print(
            f"\nIP: {result['ip']}"
        )

        print(
            f"Classification: "
            f"{result['classification']}"
        )

        print(
            f"Explanation: "
            f"{result['explanation']}"
        )

    # --------------------------------------------------------------
    # SEVERITY TESTS
    # --------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("4. SECURITY SEVERITY CLASSIFIER")
    print("=" * 70)

    incidents = [
        "User received a suspicious email.",
        "Multiple failed login attempts detected.",
        "Ransomware encrypted files on the endpoint."
    ]

    for incident in incidents:

        result = classify_severity(
            incident
        )

        print(
            f"\nIncident: {incident}"
        )

        print(
            f"Severity: {result['severity']}"
        )

    # --------------------------------------------------------------
    # FINAL
    # --------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("SECURITY TOOLS TEST COMPLETED")
    print("=" * 70)

    print(
        "\nStatus: SUCCESS"
    )

    print("\n" + "=" * 70)


# ======================================================================
# PROGRAM ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()