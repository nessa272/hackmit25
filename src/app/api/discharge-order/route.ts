import OpenAI from 'openai';
import { NextRequest, NextResponse } from 'next/server';
import { z } from 'zod';
import { zodResponseFormat } from 'openai/helpers/zod';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

const DischargeOrderSchema = z.object({
  patientName: z.string().describe("Patient's full name"),
  medicalRecordNumber: z.string().describe("Patient's medical record number"),
  dateOfBirth: z.string().describe("Patient's date of birth (MM/DD/YYYY)"),
  dischargeDate: z.string().describe("Date of discharge (MM/DD/YYYY)"),
  admissionDate: z.string().describe("Date of admission (MM/DD/YYYY)"),
  primaryDiagnosis: z.string().describe("Primary diagnosis with ICD-10 code"),
  secondaryDiagnoses: z.array(z.string()).describe("Secondary diagnoses with ICD-10 codes"),
  medications: z.array(z.object({
    name: z.string().describe("Generic medication name"),
    dosage: z.string().describe("Dosage strength (e.g., 10mg, 5ml)"),
    frequency: z.string().describe("Dosing frequency (e.g., twice daily, every 8 hours)"),
    duration: z.string().describe("Duration of treatment"),
    instructions: z.string().describe("Special administration instructions")
  })).describe("Discharge medications with complete details"),
  followUpInstructions: z.array(z.string()).describe("Follow-up appointments and care instructions"),
  activityRestrictions: z.string().describe("Physical activity limitations and restrictions"),
  dietInstructions: z.string().describe("Dietary recommendations and restrictions"),
  warningSignsToReturn: z.array(z.string()).describe("Warning signs requiring immediate medical attention"),
  dischargingPhysician: z.string().describe("Discharging physician name and credentials"),
  dischargeDisposition: z.string().describe("Discharge destination (home, SNF, rehabilitation, etc.)"),
  vitalSignsAtDischarge: z.object({
    bloodPressure: z.string().describe("Blood pressure reading"),
    heartRate: z.string().describe("Heart rate in BPM"),
    temperature: z.string().describe("Body temperature"),
    oxygenSaturation: z.string().describe("Oxygen saturation percentage"),
    respiratoryRate: z.string().describe("Respiratory rate per minute")
  }).describe("Vital signs at time of discharge"),
  functionalStatus: z.string().describe("Patient's functional capacity and mobility status"),
  dischargeInstructions: z.object({
    woundCare: z.string().nullable().describe("Wound care instructions if applicable, null if not needed"),
    equipmentNeeded: z.array(z.string()).describe("Medical equipment needed at home"),
    emergencyContacts: z.array(z.string()).describe("Emergency contact numbers"),
    transportationArrangements: z.string().describe("How patient will get home")
  }).describe("Additional discharge instructions and arrangements")
});

export async function POST(req: NextRequest) {
  try {
    const { analyses } = await req.json();
    
    if (!analyses || analyses.length === 0) {
      return NextResponse.json({ error: 'No analyses provided' }, { status: 400 });
    }

    const analysesText = analyses.map((analysis: any) => 
      `${analysis.type}: ${analysis.analysis}`
    ).join('\n\n');

    const completion = await openai.chat.completions.parse({
      model: "gpt-4o-2024-08-06",
      messages: [
        {
          role: "system",
          content: "You are an experienced attending physician creating a comprehensive discharge order. Generate realistic, detailed, and medically accurate discharge information using proper medical terminology, standard practices, and appropriate ICD-10 codes. Ensure all vital signs are within realistic ranges and all instructions are clinically appropriate."
        },
        {
          role: "user",
          content: `Create a detailed discharge order based on these medical analyses:\n\n${analysesText}`
        }
      ],
      response_format: zodResponseFormat(DischargeOrderSchema, "discharge_order"),
    });

    const dischargeOrder = completion.choices[0].message.parsed;
    
    return NextResponse.json({ dischargeOrder });
  } catch (error) {
    console.error('Discharge order generation error:', error);
    return NextResponse.json({ error: 'Discharge order generation failed' }, { status: 500 });
  }
}
