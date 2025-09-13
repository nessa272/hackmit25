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

    const analysesText = analyses.map((analysis: any) => 
      `${analysis.type}: ${analysis.analysis}`
    ).join('\n\n');

    const completion = await openai.chat.completions.create({
      model: "gpt-5-mini",
      messages: [
        {
          role: "system",
          content: "You are a discharge planning specialist. Based on the provided medical document analyses, determine if the patient is ready for discharge. Provide a clear recommendation with supporting reasoning in paragraph form only. Output plain text without any formatting, bullet points, lists, or special characters."
        },
        {
          role: "user",
          content: `Based on these medical document analyses, is this patient ready for discharge?\n\n${analysesText}`
        }
      ],
      max_completion_tokens: 3000,
    });

    return NextResponse.json({ 
      synthesis: completion.choices[0].message.content,
    });
  } catch (error) {
    return NextResponse.json({ error: 'Synthesis failed' }, { status: 500 });
  }
}
