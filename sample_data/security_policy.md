# Acme Corp Information Security Policy

Internal use only. Owner: Security Engineering. Review cycle: annual.

## Passwords and Authentication

All accounts must use passwords of at least 12 characters containing a mix of
letters, numbers and symbols. Multi-factor authentication (MFA) is mandatory
for email, VPN and all production systems. Shared accounts are prohibited.

## Network Access

The corporate VPN is mandatory for any remote access to internal systems,
including the data warehouse and the admin consoles. Split tunnelling is
disabled by policy. Personal devices may only reach the guest network.

## Data Classification

Data is classified into four levels: Public, Internal, Confidential and
Restricted. Restricted data (credentials, payment data, health records) must
be encrypted at rest and in transit and may never be copied to personal
devices or unmanaged cloud storage.

## Incident Reporting

Suspected security incidents must be reported to the security team within
24 hours of discovery via the #security-incidents channel or
security@acme.example. Do not attempt to investigate or remediate on your own
before reporting.

## Data Retention

Financial records are retained for 7 years. Application logs are retained for
90 days. Customer personal data is deleted within 30 days of a verified
deletion request.
