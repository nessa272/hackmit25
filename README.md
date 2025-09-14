# Healthcare Discharge Planning AI System

An intelligent healthcare application that leverages fine-tuned AI models to analyze medical documents and provide evidence-based discharge recommendations for hospital patients.

## Overview

This system processes multiple types of medical documentation to generate comprehensive discharge assessments, helping healthcare teams make informed decisions about patient readiness for discharge while identifying potential risks and safety considerations.

## Features

### Multi-Document Analysis
- **Medical Reports**: Analyzes diagnostic procedures, lab results, and clinical findings
- **Progress Notes**: Reviews patient status updates and treatment responses  
- **Nurse Rounding**: Processes vital signs, medication administration, and patient observations
- **Insurance Documentation**: Handles coverage and authorization information

### AI-Powered Clinical Reasoning
- **Fine-tuned GPT Model**: Custom healthcare model trained on discharge prediction data
- **Intelligent Data Extraction**: Converts unstructured medical text into structured clinical data
- **Risk Assessment**: Evaluates discharge readiness with severity classification (Low/Moderate/High/Critical)

### Comprehensive Discharge Assessment
- **Discharge Readiness Factors**: Evidence-based reasons supporting patient discharge
- **Key Risk Identification**: Potential complications and safety concerns
- **Clinical Reasoning**: AI-generated analysis considering patient history and current status
- **Structured Recommendations**: Clear, actionable discharge decisions

### Automated Documentation
- **Discharge Order Generation**: Creates complete, formatted discharge orders
- **PDF Export**: Professional discharge documentation for medical records
- **Medication Lists**: Detailed prescriptions with dosages and instructions

## Technology Stack

- **Frontend**: Next.js 14 with TypeScript and Tailwind CSS
- **AI Integration**: 
  - Fine-tuned GPT-OSS-120B model (Modal deployment)
  - OpenAI GPT-4o-mini for data extraction
- **Document Processing**: Multi-format file upload and analysis
- **Infrastructure**: Modal cloud platform for model serving

## Getting Started

### Prerequisites
- Node.js 18+ 
- Python 3.8+ (for fine-tuning module)
- OpenAI API key
- Modal account and API key for fine-tuned model access
- Cerebras API key for enhanced AI processing

### Installation

See [SETUP.md](SETUP.md) for detailed installation instructions.

**Quick Start:**
```bash
# Clone and setup
git clone <repository-url>
cd Hackmit25
npm install

# Configure environment
# Create .env.local with your API keys

# Run application
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to access the application

### Requirements Files
- `package.json` - Node.js dependencies
- `requirements.txt` - Python dependencies (main)
- `finetuned/requirements.txt` - Fine-tuning module dependencies

## Usage

1. **Upload Documents**: Add medical reports, progress notes, and nurse rounding documents
2. **AI Analysis**: The system automatically processes each document type
3. **Clinical Reasoning**: Fine-tuned model generates healthcare-specific insights
4. **Discharge Assessment**: Comprehensive recommendation appears at the top of results
5. **Generate Orders**: Create formal discharge documentation with one click

## Fine-Tuned Model

The system uses a custom GPT-OSS-120B model fine-tuned on healthcare discharge data:
- **Training Data**: 46 healthcare records with discharge outcomes
- **Model Performance**: 50% improvement in discharge prediction accuracy
- **Deployment**: Modal cloud infrastructure with H100 GPU support

## API Endpoints

- `/api/medical-reports` - Process medical diagnostic reports
- `/api/progress-notes` - Analyze patient progress documentation  
- `/api/nurse-rounding` - Process nursing observations and vital signs
- `/api/patient-reasoning` - Generate AI clinical reasoning
- `/api/synthesizer` - Create comprehensive discharge assessment
- `/api/discharge-order` - Generate formal discharge documentation

## Contributing

This project was developed for healthcare workflow optimization. Contributions should focus on improving clinical accuracy and user experience for healthcare professionals.

## License

Built for HackMIT 2025 - Healthcare AI Track
