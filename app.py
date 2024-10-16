from flask import Flask, render_template, request, send_file, flash, redirect, session
from pymed import PubMed
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from diffusers import StableDiffusionPipeline
import torch
import logging
from fpdf import FPDF
from flask_caching import Cache

# Initialize Flask App
app = Flask(__name__)
app.secret_key = "supersecretkey"  # Necessary for session management

# Logging setup
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Configure Flask-Cache
cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache'})

# PubMed setup
pubmed = PubMed(tool="MyMedicalApp", email="my@email.address")

# Load models during startup
print("Loading models... This may take a while.")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load BioGPT for medical Q&A
tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT")
model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT").to(device)
model.config.pad_token_id = tokenizer.eos_token_id

# Define medical Q&A pipeline
medical_qa = pipeline(
    "text-generation", model=model, tokenizer=tokenizer,
    max_length=300, truncation=True, device=0 if torch.cuda.is_available() else -1
)

# Load Stable Diffusion for image generation
sd_pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)


@app.route("/", methods=["GET", "POST"])
def index():
    """Home route with search bar."""
    return render_template("index.html")


@app.route("/search", methods=["GET"])
def search():
    """Search route to handle PubMed queries."""
    query = request.args.get("query", "").strip()
    if not query:
        flash("Please enter a valid query.", "warning")
        return redirect("/")

    try:
        # Perform PubMed search
        results = pubmed.query(query, max_results=10)
        articles = []

        # Parse results and prepare them for rendering
        for article in results:
            title = article.title
            authors = ", ".join([author['lastname'] for author in article.authors])
            pub_date = article.publication_date
            abstract = article.abstract
            link = f"https://pubmed.ncbi.nlm.nih.gov/{article.pubmed_id}/"

            articles.append({
                "title": title,
                "authors": authors,
                "date": pub_date,
                "abstract": abstract,
                "link": link
            })

        return render_template("search_results.html", query=query, articles=articles)
    except Exception as e:
        logging.error(f"PubMed Search Error: {str(e)}")
        flash("Failed to fetch search results. Please try again.", "danger")
        return redirect("/")


@app.route("/symptoms", methods=["GET", "POST"])
def symptoms_page():
    """Symptoms page where the user can input symptoms."""
    if request.method == "POST":
        symptoms = request.form.get("symptoms")
        if not symptoms.strip():
            flash("Please provide symptoms for diagnosis.", "warning")
            return redirect("/symptoms")

        # Check cache for existing diagnosis
        cached_response = cache.get(symptoms)
        if cached_response:
            return render_template("symptoms.html", response=cached_response)

        prompt = f"Symptoms: {symptoms}\n\nPossible Diagnosis: "
        session["conversation"] = session.get("conversation", [])
        session["conversation"].append(prompt)
        context = " ".join(session["conversation"])

        try:
            # Generate diagnosis
            result = medical_qa(context)[0]['generated_text']
            
            # Cache the result
            cache.set(symptoms, result)

            # Clear GPU memory
            torch.cuda.empty_cache()

            return render_template("symptoms.html", response=result)
        except Exception as e:
            logging.error(f"Diagnosis Error: {str(e)}")
            flash("Error generating diagnosis. Please try again.", "danger")
            return render_template("symptoms.html")

    return render_template("symptoms.html")


@app.route("/medimage", methods=["GET", "POST"])
def medimage_page():
    """MedImage Generator page."""
    if request.method == "POST":
        prompt = request.form.get("prompt")
        if not prompt.strip():
            flash("Please provide a prompt for the image generation.", "warning")
            return redirect("/medimage")

        try:
            # Generate image
            image = sd_pipeline(prompt).images[0]
            image_path = "static/generated_image.png"
            image.save(image_path)

            logging.info(f"Image Generated for prompt: {prompt}")
            return render_template("medimage.html", image_url=image_path)
        except Exception as e:
            logging.error(f"Image Generation Error: {str(e)}")
            flash("Failed to generate the image. Please try again.", "danger")
            return render_template("medimage.html")

    return render_template("medimage.html")


@app.route("/diagnostic", methods=["GET", "POST"])
def diagnostic_page():
    """Diagnostic Report page."""
    if request.method == "POST":
        patient_data = request.form.get("patient_data")
        if not patient_data.strip():
            flash("Please provide patient data.", "warning")
            return redirect("/diagnostic")

        prompt = f"Patient Data: {patient_data}\n\nTreatment Plan: "

        try:
            # Generate treatment plan
            result = medical_qa(prompt)[0]['generated_text']

            # Clear GPU memory
            torch.cuda.empty_cache()

            logging.info(f"Patient Data: {patient_data}, Treatment Plan: {result}")
            return render_template("diagnostic.html", response=result)
        except Exception as e:
            logging.error(f"Treatment Plan Error: {str(e)}")
            flash("Error generating treatment plan. Please try again.", "danger")
            return render_template("diagnostic.html")

    return render_template("diagnostic.html")


@app.route("/download", methods=["POST"])
def download_report():
    """Generate and download a diagnosis report as PDF."""
    diagnosis = request.form.get("diagnosis")
    if not diagnosis.strip():
        flash("Diagnosis is required for the report.", "warning")
        return redirect("/")

    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Diagnosis Report", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, txt=f"Diagnosis: {diagnosis}")

        pdf_path = "static/diagnosis_report.pdf"
        pdf.output(pdf_path)

        logging.info("PDF Report generated successfully.")
        return send_file(pdf_path, as_attachment=True)
    except Exception as e:
        logging.error(f"PDF Generation Error: {str(e)}")
        flash("Failed to generate report. Please try again.", "danger")
        return redirect("/")


@app.errorhandler(Exception)
def handle_exception(e):
    """Global error handler."""
    logging.error(f"Unhandled Exception: {str(e)}")
    flash("An unexpected error occurred. Please try again later.", "danger")
    return render_template("index.html"), 500


if __name__ == "__main__":
    app.run(debug=True)
