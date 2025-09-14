import OpenAI from 'openai';
import { NextRequest, NextResponse } from 'next/server';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

export async function POST(req: NextRequest) {
  try {
    const { analyses, reasoning } = await req.json();
    
    if (!analyses || analyses.length === 0) {
      return NextResponse.json({ error: 'No analyses provided' }, { status: 400 });
    }

    const analysesText = analyses.map((analysis: any) => 
      `${analysis.type}: ${analysis.analysis}`
    ).join('\n\n');

    const reasoningText = reasoning?.reasoning || '';
    const reasoningSource = reasoning?.source || 'not available';

    const completion = await openai.chat.completions.create({
      model: "gpt-5-mini",
      messages: [
        {
          role: "system",
          content: "You are a discharge planning specialist. Provide a BRIEF discharge assessment emphasizing CONCISENESS. Give PRIMARY EMPHASIS to the clinical reasoning from the fine-tuned model. Use bullet points with these sections: 1) Discharge Readiness (max 2 bullets, 1 sentence each), 2) Key Risks (max 2 bullets, 1 sentence each), 3) Risk Level (1 word: Low/Moderate/High/Critical), 4) Recommendation (1 sentence). Be extremely concise - total response should be under 300 words."
        },
        {
          role: "user",
          content: `Provide a BRIEF discharge assessment that details whether or not the patient is ready to be discharged and why. Please keep this response under 400 words.

                    CLINICAL REASONING (${reasoningSource}):
                    ${reasoningText}

                    MEDICAL DOCUMENT ANALYSES:
                    ${analysesText}

                    When assessing discharge readiness, consider the clinical reasoning and take the recommendation into account by calculating the amount of time the patient has ALREADY stayed in the hospital (based on the analyses), and thinking about the recommendation.

                    Format as:
                    • Discharge Readiness: (max 2 bullets, 1 sentence each)
                    • Key Risks: (max 2 bullets, 1 sentence each)  
                    • Risk Level: (1 word only)
                    • Recommendation: (1 sentence only)
                    • Reasoning: (one paragraph only)`
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
