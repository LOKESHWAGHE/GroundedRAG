# Acme Robotics Fleet API Integration Guide

## Authentication

The Fleet API uses OAuth 2.0 client credentials flow. Tokens are issued by
the endpoint `https://auth.acmerobotics.example/oauth/token` and expire
after 3600 seconds. Clients must refresh the token before expiry; there is
no refresh token grant supported.

## Rate Limits

The API enforces a rate limit of 120 requests per minute per API key.
Exceeding this limit returns an HTTP 429 response with a `Retry-After`
header indicating the number of seconds to wait.

## Webhook Events

The Fleet API can send webhook notifications for the following events:
`arm.fault`, `arm.task_completed`, `arm.maintenance_due`, and
`arm.offline`. Webhook payloads are signed with HMAC-SHA256 using a
per-account secret key.

## Supported Regions

The Fleet API is currently available in the ap-south-1 (Mumbai) and
ap-southeast-1 (Singapore) regions. Requests from unsupported regions are
rejected with an HTTP 403 response.
