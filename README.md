Medical AI Web App Documentation
Introduction
This project is a Flask-based web application that integrates AI-driven features for medical search, diagnosis generation, and image creation. The app leverages PyMed for accessing PubMed articles, Microsoft's BioGPT for medical question-answering, and Stable Diffusion for generating medical-related images based on text prompts.
Features
- Search PubMed articles based on user queries
- Generate medical diagnoses based on user-reported symptoms
- Create medical-related images using Stable Diffusion
- Download diagnosis reports as PDF
Requirements
To run this project on a new device, ensure you have the following installed:
1. Python 3.8+
2. Flask
3. PyMed
4. Huggingface Transformers
5. Diffusers
6. FPDF for PDF generation
7. torch (with CUDA support for GPU acceleration)
8. Flask-Caching
Installation Steps
1. Clone the repository or download the code files.
2. Set up a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For Linux/MacOS
   venv\Scripts\activate   # For Windows
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure CUDA drivers are installed and compatible if using GPU.
5. Set environment variables for Flask app secret key (optional for security).
6. Run the application locally:
   ```bash
   flask run
   ```
Configuration
Ensure the following configurations are updated based on your environment:
- `secret_key` in the Flask app is set to a strong value.
- CUDA support is correctly enabled for GPU acceleration if using it.
- Update the email address in the PubMed setup to your own for API rate limits.
Usage Instructions
1. Access the application at http://127.0.0.1:5000/ once it is running.
2. Use the navigation to access the following features:
   - Search: Enter medical queries to search PubMed articles.
   - Symptoms: Enter symptoms to get a possible diagnosis.
   - MedImage: Provide a text prompt to generate a medical-related image.
   - Diagnostic: Input patient data to generate a treatment plan.
   - Download: Generate a PDF report based on the diagnosis.

Troubleshooting
1. Ensure all Python packages are correctly installed.
2. If using GPU, ensure CUDA is installed and supported.
3. Review the logs (`app.log`) for detailed error messages.
Conclusion
This Medical AI Web App demonstrates the power of integrating modern AI techniques into a web-based platform, providing users with helpful tools for research, diagnosis, and medical imaging.
