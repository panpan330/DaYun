# Knowledge Base Sample Documents

This directory contains small mock knowledge-base documents for RAG learning.

The documents are intentionally simple and controllable. They are not real company policies.

Current documents:

```text
order-shipping-policy.md      Order shipping rules and delayed shipment handling.
refund-return-policy.md       Refund and return rules.
logistics-tracking-faq.txt    Logistics tracking FAQ in plain text format.
account-security-faq.md       Account security and password reset FAQ.
```

Suggested metadata for later lessons:

```text
source              File path or file name.
title               Document title.
doc_type            policy or faq.
business_domain     order, refund, logistics, or account.
permission_group    customer_service.
```

These files will be used in later lessons for:

```text
document loading
text cleaning
chunk splitting
embedding generation
Qdrant point payload design
retrieval testing
source citation
```
