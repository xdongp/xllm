# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in xLLM, please report it to us responsibly.

### How to Report

1. **Do not** open a public issue
2. Send an email to: security@xllm.dev
3. Include as much detail as possible:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if known)

### What to Expect

- We will acknowledge receipt of your report within 48 hours
- We will provide a detailed response within 7 days
- We will work with you to understand and resolve the issue
- We will notify you when the fix is released
- We will credit you in the release notes (if you wish)

### Security Best Practices

When using xLLM, please follow these security best practices:

1. **Model Security**
   - Only use models from trusted sources
   - Verify model checksums before use
   - Keep model files in secure directories

2. **API Security**
   - Use HTTPS in production
   - Implement proper authentication
   - Rate limit API endpoints
   - Validate all user inputs

3. **Network Security**
   - Do not expose the server to public internet without proper security measures
   - Use firewalls to restrict access
   - Keep dependencies updated

4. **Data Privacy**
   - Be aware that prompts are sent to the model
   - Do not send sensitive information
   - Implement proper data retention policies

## Security Updates

We will announce security updates through:
- GitHub Security Advisories
- Release notes
- Email notifications (for critical vulnerabilities)

## Dependencies

We regularly audit and update dependencies to ensure security. Please keep your installation up to date:

```bash
pip install --upgrade -r requirements.txt
```

## Security Hall of Fame

We would like to thank everyone who has helped make xLLM more secure by reporting vulnerabilities responsibly.

*If you have reported a vulnerability and would like to be credited here, please let us know.*
