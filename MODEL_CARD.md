# Model Card: Customer Inquiry Classifier

## Model Details

- Task: Multi-class text classification for customer-support routing.
- Model family: Classical NLP pipeline.
- Features: TF-IDF over normalized customer inquiry text.
- Estimators: Calibrated LinearSVC and Logistic Regression in a soft-voting ensemble.
- Output: Category, confidence, routing decision, destination queue, and keyword-style explanation.

## Intended Use

Use this model for first-pass routing of customer-support messages into operational queues such as billing, technical support, product inquiry, shipping, refund/return, account management, and general inquiry.

The model should not be the final authority for sensitive support actions. Low-confidence cases are intentionally escalated to human review.

## Training Data

The current training data is generated synthetically by `DataGenerator` in `app/classifier.py`. It includes category-specific templates, noisy text, ambiguous examples, and tone variation.

## Evaluation Guidance

Track these metrics when refreshing the model:

- Accuracy for broad sanity checks.
- Macro F1 to avoid hiding weak minority classes.
- Per-class precision and recall.
- Confusion matrix between similar categories.
- Human-review rate at the selected confidence threshold.
- Latency for single and batch inference.

## Limitations

- Synthetic data cannot fully represent real customer language.
- The model may struggle with multilingual, code-mixed, or highly domain-specific messages.
- Confidence calibration should be rechecked after changing categories or training data.
- Optional LLM fallback adds cost, latency, and external dependency risk.

## Responsible Use

Use the confidence router. Do not auto-close support tickets based only on this model. Review escalated or low-confidence cases and collect anonymized real examples before production use.
