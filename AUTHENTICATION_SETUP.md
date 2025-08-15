# Authentication Setup Guide

## Overview

This dashboard now includes password protection to secure access to your STT evaluation data. The authentication system supports multiple deployment scenarios with different security levels.

## Security Features

- ✅ Password-based authentication
- ✅ Session-based login (stays logged in during session)
- ✅ Secure password hashing (SHA-256)
- ✅ Logout functionality
- ✅ Professional login interface
- ✅ Support for environment variables and Streamlit secrets

## Setup Instructions

### For Streamlit Cloud Deployment (Recommended)

1. **Deploy your app** to Streamlit Cloud as usual

2. **Configure secrets** in your Streamlit Cloud app:
   - Go to your app settings in Streamlit Cloud
   - Navigate to the "Secrets" section
   - Add the following configuration:
   ```toml
   STREAMLIT_PASSWORD = "your_secure_password_here"
   ```
   - Replace `"your_secure_password_here"` with your desired password
   - Save the configuration

3. **Restart your app** for the changes to take effect

### For Local Development

The app will use the fallback password `stt_dashboard_2024` by default. You can:

1. **Change the fallback password** in `evaluation_dashboard.py`:
   ```python
   DASHBOARD_PASSWORD = os.getenv("STREAMLIT_PASSWORD", "your_local_password")
   ```

2. **Or set an environment variable**:
   ```bash
   export STREAMLIT_PASSWORD="your_local_password"
   streamlit run evaluation_dashboard.py
   ```

### For Other Deployment Platforms

Set the `STREAMLIT_PASSWORD` environment variable in your deployment platform:

- **Heroku**: Use config vars
- **Docker**: Use environment variables in your container
- **AWS/GCP/Azure**: Use their respective secret management services

## Password Security Best Practices

1. **Use a strong password**:
   - At least 12 characters long
   - Include uppercase, lowercase, numbers, and special characters
   - Avoid common words or patterns

2. **Don't commit passwords to your repository**:
   - The `.streamlit/secrets.toml` file is provided as a template
   - Always use secrets management for production

3. **Regularly rotate passwords**:
   - Update the password periodically
   - Restart the app after changing the password

## Usage

1. **Access the dashboard** - users will see a login screen
2. **Enter the password** - the interface is user-friendly with clear instructions
3. **Navigate the dashboard** - full access to all features after authentication
4. **Logout** - use the logout button in the sidebar to end the session

## Troubleshooting

### Common Issues

1. **"Password incorrect" message**:
   - Double-check the password in your secrets/environment variables
   - Ensure there are no extra spaces or characters
   - Restart the app after making changes

2. **Login form not appearing**:
   - Check the browser console for errors
   - Ensure all dependencies are installed
   - Verify the app is running the latest version of the code

3. **Secrets not working in Streamlit Cloud**:
   - Verify the secrets are properly formatted in the Streamlit Cloud interface
   - Check that the key name matches exactly: `STREAMLIT_PASSWORD`
   - Restart the app after updating secrets

### Security Considerations

- **Session-based**: Authentication persists only for the browser session
- **No persistent storage**: Passwords are not stored in the app
- **Hashed comparison**: Passwords are hashed before comparison
- **Automatic logout**: Users can logout manually or session expires when browser closes

## Alternative Authentication Methods

If you need more advanced authentication, consider:

1. **streamlit-authenticator**: Full-featured authentication library
2. **OAuth integration**: Google, GitHub, etc.
3. **LDAP/Active Directory**: Enterprise authentication
4. **Multi-factor authentication**: Additional security layer

For now, the simple password-based system provides good security for most use cases while being easy to deploy and manage.

## Support

If you encounter issues with the authentication setup, check:
1. Streamlit Cloud logs for error messages
2. Browser developer console for client-side errors
3. Ensure all environment variables/secrets are properly configured 