# Dataset Card: Synthetic Customer Inquiries

## Dataset Summary

The dataset is generated in code by `DataGenerator` rather than downloaded from a public source. It simulates customer-support messages across seven categories:

- billing
- technical_support
- product_inquiry
- shipping
- refund_return
- account_management
- general_inquiry

## Why Synthetic Data

Synthetic data avoids exposing private customer information and makes the project reproducible for recruiters and reviewers. It is useful for demonstrating the ML pipeline, API design, and confidence-routing behavior.

## Generation Approach

The generator creates examples with:

- category-specific message templates
- short and long wording variants
- urgency and sentiment variation
- typos/noise
- ambiguous messages that should be escalated more often

## Known Gaps

- It does not guarantee real-world distribution.
- It may underrepresent regional language, slang, sarcasm, and multi-intent tickets.
- Performance on synthetic data should not be presented as production performance on real users.

## Recommended Next Dataset Step

Add a small anonymized validation set from real or manually written support tickets, then compare metrics between synthetic validation and realistic validation.
