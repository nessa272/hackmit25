# Setup Guide - Healthcare Discharge Planning AI System

## System Requirements

### Hardware Requirements
- **Minimum**: 8GB RAM, 4-core CPU
- **Recommended**: 16GB+ RAM, 8-core CPU
- **GPU**: Optional for local development, required for model training
- **Storage**: 10GB+ free space

### Software Requirements
- **Node.js**: Version 18.0 or higher
- **Python**: Version 3.8 or higher (for fine-tuning module)
- **npm**: Version 8.0 or higher
- **Git**: Latest version

## API Keys Required

### 1. OpenAI API Key
- **Purpose**: Powers GPT-4o-mini for medical data extraction
- **How to get**: Visit [OpenAI Platform](https://platform.openai.com/api-keys)
- **Pricing**: Pay-per-use, ~$0.001 per 1K tokens

### 2. Modal API Key
- **Purpose**: Serves fine-tuned GPT-OSS-120B model
- **How to get**: Sign up at [Modal](https://modal.com) and generate API key
- **Pricing**: GPU-based pricing, H100 ~$4/hour

### 3. Cerebras API Key
- **Purpose**: Enhanced AI processing capabilities
- **How to get**: Register at [Cerebras Cloud](https://cloud.cerebras.ai)
- **Pricing**: Token-based pricing

### 4. HuggingFace Token (Optional)
- **Purpose**: Access to model repositories
- **How to get**: Create account at [HuggingFace](https://huggingface.co/settings/tokens)
- **Pricing**: Free for public models

## Installation Steps

### 1. Clone Repository
```bash
git clone <repository-url>
cd Hackmit25
```

### 2. Frontend Setup
```bash
# Install Node.js dependencies
npm install

# Create environment file
touch .env.local

# Edit .env.local with your API keys
nano .env.local
```

### 3. Python Environment Setup (Optional - for fine-tuning)
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# For fine-tuning module specifically
cd finetuned
pip install -r requirements.txt
```

### 4. Modal Setup (Required for fine-tuned model)
```bash
# Install Modal CLI
pip install modal

# Authenticate with Modal
modal token new

# Deploy fine-tuned model (if needed)
cd finetuned
modal deploy gpt_oss_120b_finetune.py
```

## Configuration

### Environment Variables
Edit `.env.local` with your API keys:
```bash
OPENAI_API_KEY=sk-...
MODAL_API_KEY=ak-...
CEREBRAS_API_KEY=csk-...
HF_TOKEN=hf_...
```

### Model Endpoints
The system expects these Modal endpoints to be available:
- Fine-tuned model: `https://nessa272--predict-los-dev.modal.run`
- Health check: `https://nessa272--health-check-dev.modal.run`
- Model info: `https://nessa272--models-dev.modal.run`

## Running the Application

### Development Mode
```bash
npm run dev
```
Access at: http://localhost:3000

### Production Build
```bash
npm run build
npm start
```

## Troubleshooting

### Common Issues

#### 1. API Key Errors
- Verify all API keys are correctly set in `.env.local`
- Check API key permissions and quotas
- Ensure no extra spaces or quotes around keys

#### 2. Modal Connection Issues
- Verify Modal token: `modal token current`
- Check model deployment status
- Ensure correct endpoint URLs

#### 3. Document Upload Failures
- Supported formats: PDF, TXT, DOC, DOCX
- Maximum file size: 10MB per file
- Check file permissions and encoding

#### 4. Slow Response Times
- Fine-tuned model cold starts can take 30-60 seconds
- Consider upgrading Modal plan for faster GPU access
- Monitor API rate limits

### Performance Optimization

#### 1. Model Serving
- Use Modal's auto-scaling for production
- Consider model caching strategies
- Monitor GPU utilization

#### 2. Frontend Performance
- Enable Next.js production optimizations
- Use CDN for static assets
- Implement proper loading states

## Security Considerations

### API Key Management
- Never commit API keys to version control
- Use environment variables for all secrets
- Rotate keys regularly
- Monitor API usage for anomalies

### Data Privacy
- Medical data is processed in memory only
- No persistent storage of patient information
- All API calls use HTTPS encryption
- Consider HIPAA compliance requirements

## Support

### Documentation
- Next.js: https://nextjs.org/docs
- Modal: https://modal.com/docs
- OpenAI: https://platform.openai.com/docs

### Common Commands
```bash
# Check system status
npm run dev

# View logs
modal logs <deployment-name>

# Test API endpoints
curl -X POST http://localhost:3000/api/medical-reports

# Clear cache
rm -rf .next node_modules
npm install
```
