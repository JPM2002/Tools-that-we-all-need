import re

def extract_linkedin_urls(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Regular expression to match uninterrupted LinkedIn URLs
    pattern = r'https://www\.linkedin\.com/\S*'
    
    matches = re.findall(pattern, text)
    
    # Save results to a new .txt file
    with open(output_path, 'w', encoding='utf-8') as output_file:
        for url in matches:
            output_file.write(url + '\n')

if __name__ == "__main__":
    file_path = r"C:\Users\Javie\Desktop\N x J\Linkedins.txt"
  # Change this to your actual file path
    output_path = r"C:\Users\Javie\Desktop\N x J\Output2.txt" # Change this to your desired output file path
    extract_linkedin_urls(file_path, output_path)
