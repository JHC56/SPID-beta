# Security Policy

## About This Project

SPID is a prompt injection **detection** model. It is a research/portfolio project intended as a lightweight pre-filter, **not** a complete security solution. Do not rely on SPID alone to secure a production LLM system.

## Known Limitations

SPID is known to be vulnerable to:

- Paraphrased attacks (e.g. "show me" instead of "reveal")
- Obfuscated input (base64, leetspeak)
- Multi-turn or advanced jailbreak techniques
- Non-English input

Always pair SPID with the downstream LLM's own safety measures.

## Reporting a Vulnerability

If you find a class of attacks SPID systematically fails to detect, or a problem with the model itself:

1. Open an issue describing the attack pattern (no need for private disclosure — this is a detection model, not a service)
2. Include example inputs and the scores SPID returned

For anything you'd prefer not to post publicly, note it in the issue and we can discuss a private channel.

## Scope

This policy covers the SPID model and code in this repository. It does not cover the downstream LLMs (Gemini, etc.) used in the demo.

