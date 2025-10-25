import secrets

def generate_jwt_secret():
    """Generate a secure random string for JWT secret key."""
    return secrets.token_hex(32)

if __name__ == "__main__":
    secret = generate_jwt_secret()
    print("\nGenerated JWT Secret Key:")
    print("------------------------")
    print(secret)
    print("\nAdd this to your .env file as:")
    print(f'JWT_SECRET_KEY={secret}')