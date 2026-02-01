import sys
import qrcode


def validate_url(url):
    """
    Basic URL validation to check if URL is empty or valid.
    """
    if not url or not url.strip():
        return False
    
    url = url.strip()
    # http:// or https:// checks
    if url.startswith(('http://', 'https://')):
        return True
    # contain a dot checks
    if '.' in url:
        return True
    
    return False


def generate_qr_code(url, output_filename='qrcode.png'):
    """
    Generate QR code from URL and save file.
    
    Args:
        url (str): The URL for QR code
        output_filename (str): filename to save the QR code image
    
    Returns:
        string: The path to the saved QR code image file
    """
    try:
        # Create QR code instance
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,  # Error correction level
            box_size=10,
            border=4,  # Border thickness
        )
        
        # Add data to QR code
        qr.add_data(url)
        qr.make(fit=True)
        
        # Create image from QR code
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Save the image
        img.save(output_filename)
        
        return output_filename
    
    except Exception as e:
        print(f"Error generating QR code: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main function"""
    
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input("Enter URL: ").strip()
    
    # Validate URL
    if not validate_url(url):
        print("Error: Invalid URL!!! Please provide a valid URL (e.g., https://google.com)", file=sys.stderr)
        sys.exit(1)
    
    # Generate QR code
    output_file = generate_qr_code(url)
    
    # Print success message with file path
    print(f"QR code generated successfully!")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
