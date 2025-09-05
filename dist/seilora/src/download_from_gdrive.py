import requests
import re
from pathlib import Path
import logging
import urllib.parse
from typing import Dict, Tuple, Literal

# Set up a basic logger for a clean output
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_form_details(html_content: str) -> Tuple[str, str, Dict[str, str]] | None:
    """
    Extracts the form action URL, method, and hidden input name-value pairs from the HTML.

    Args:
        html_content: The HTML content of the warning page.

    Returns:
        A tuple containing the action URL, method, and a dictionary of form parameters,
        or None if the form is not found.
    """
    # Look for the form that handles the download submission and capture its action and method
    form_pattern = re.search(r'<form\s+id="download-form"\s+action="([^"]+)"\s+method="([^"]+)"', html_content)
    if not form_pattern:
        return None
    
    action_url = form_pattern.group(1)
    method = form_pattern.group(2).upper()
    
    # Extract all hidden input fields from the form
    input_pattern = re.findall(r'<input\s+type="hidden"\s+name="([^"]+)"\s+value="([^"]+)"', html_content)
    
    form_params = dict(input_pattern) if input_pattern else {}
    
    return action_url, method, form_params

def download_from_gdrive(file_id: str, destination: Path):
    """
    Downloads a file from Google Drive by using a form-based approach
    to handle the virus scan warning.

    Args:
        file_id: The unique ID of the Google Drive file.
        destination: The local path to save the downloaded file.
    """
    session = requests.Session()
    
    # A standard User-Agent header
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    
    initial_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    try:
        # Step 1: Request the page to get the form data
        response = session.get(initial_url, headers=headers)
        response.raise_for_status()

        # Step 2: Check for the virus scan warning
        if 'Virus scan warning' in response.text:
            logger.info("Virus scan warning page detected. Extracting form data...")
            
            form_url, form_method, form_params = extract_form_details(response.text)
            
            if form_url and form_method and form_params:
                post_url = urllib.parse.urljoin('https://drive.google.com', form_url)

                # Step 3: Use the correct HTTP method (GET or POST) based on the form
                if form_method == 'POST':
                    final_response = session.post(post_url, data=form_params, headers=headers, stream=True)
                else: # Default to GET if not POST
                    final_response = session.get(post_url, params=form_params, headers=headers, stream=True)
                
                final_response.raise_for_status()

                # Step 4: Save the file content from the final response
                logger.info("Download request successful. Saving file...")
                with open(destination, 'wb') as f:
                    for chunk in final_response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                logger.info(f"Successfully downloaded to {destination}")
                return True
            else:
                logger.error("Could not find download form on the warning page. HTML structure may have changed.")
                return False

        # If no warning page, the initial request is the direct download
        else:
            logger.info("No virus warning detected. Direct download successful.")
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logger.info(f"Successfully downloaded to {destination}")
            return True

    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred: {e}")
        return False

# --- Main Execution ---
if __name__ == "__main__":
    target_file_id = "1MTTLYGQN_SL80MvkAhHPqPvpYkX5QCLu"
    output_path = Path("downloaded_file.pth")
    
    if not download_from_gdrive(target_file_id, output_path):
        logger.error("Download failed.")
