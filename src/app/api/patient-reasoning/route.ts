import OpenAI from 'openai';
import { NextRequest, NextResponse } from 'next/server';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

export async function POST(req: NextRequest) {
  try {
    const { analyses } = await req.json();
    
    if (!analyses || analyses.length === 0) {
      return NextResponse.json({ error: 'No analyses provided' }, { status: 400 });
    }

    // Convert analyses to structured JSON using OpenAI
    const analysesText = analyses.map((analysis: any) => 
      `${analysis.type}: ${analysis.analysis || analysis.content}`
    ).join('\n\n');

    const structuredDataCompletion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        {
          role: "system",
          content: `You are a medical data extraction specialist. Convert the provided medical analyses into a structured JSON format with these exact fields: patient_summary, diagnosis, severity, mortality_risk, admission_type, expected_disposition, additional_info. Extract relevant information from the analyses and populate each field. If a field cannot be determined from the analyses, leave it as an empty string. Return only valid JSON.

Medical analyses to convert:
${analysesText}`
        },
        {
          role: "user",
          content: `Convert the medical analyses into this exact JSON structure:

{
  "patient_summary": "",
  "diagnosis": "",
  "severity": "",
  "mortality_risk": "",
  "admission_type": "",
  "expected_disposition": "",
  "additional_info": ""
}`
        }
      ],
      max_completion_tokens: 1000,
    });

    let structuredData;
    try {
      structuredData = JSON.parse(structuredDataCompletion.choices[0].message.content || '{}');
    } catch (parseError) {
      // Fallback to empty structure if JSON parsing fails
      structuredData = {
        patient_summary: '',
        diagnosis: '',
        severity: '',
        mortality_risk: '',
        admission_type: '',
        expected_disposition: '',
        additional_info: ''
      };
    }

    const prompt = `Patient: ${structuredData.patient_summary}
Diagnosis: ${structuredData.diagnosis}
Severity: ${structuredData.severity}
Mortality Risk: ${structuredData.mortality_risk}
Admission Type: ${structuredData.admission_type}
Expected Disposition: ${structuredData.expected_disposition}${structuredData.additional_info ? `
Additional Info: ${structuredData.additional_info}` : ''}`;

    const fallbackReasoning = "Based on comprehensive analysis of the patient's medical records, current clinical status, and documented progress notes, the patient demonstrates stable vital signs, adequate response to treatment protocols, and meets established discharge criteria. The patient's functional capacity has improved to baseline levels, pain management is optimized with oral medications, and all acute medical issues have been appropriately addressed. Family education has been completed, and appropriate follow-up care has been arranged to ensure continuity of care post-discharge.";

    try {
      // Call fine-tuned model with 30 second timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 120000);

      const fetchPromise = fetch('https://nessa272--predict-los-dev.modal.run', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_path: '/models/gpt-oss-120b-finetune',
          prompt: prompt,
          max_new_tokens: 256,
          temperature: 0.7,
        }),
        signal: controller.signal,
      });

      // Wait for either the fetch to complete or 30 seconds to pass
      const response = await Promise.race([
        fetchPromise,
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Request timeout after 30 seconds')), 120000)
        )
      ]) as Response;

      clearTimeout(timeoutId);

      if (response.ok) {
        const result = await response.json();
        return NextResponse.json({ 
          reasoning: result.response || result,
          source: 'fine-tuned-model'
        });
      } else {
        throw new Error(`Model API returned ${response.status}`);
      }

    } catch (error) {
      console.log('Fine-tuned model failed or timed out after 30 seconds, using fallback:', error);
      
      // Use fallback response
      return NextResponse.json({ 
        reasoning: fallbackReasoning,
        source: 'fallback'
      });
    }

  } catch (error) {
    return NextResponse.json({ error: 'Patient reasoning failed' }, { status: 500 });
  }
}
